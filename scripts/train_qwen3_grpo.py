"""
Qwen3 Advanced GRPO Training Script
====================================
Group Relative Policy Optimization (GRPO) for Qwen3 models on Windows/Linux.

Qwen3 key features leveraged:
  - Thinking mode  : <think>…</think> reasoning chain before final answer
  - Instruct format: ChatML with system / user / assistant turns
  - LoRA fine-tuning via Unsloth's FastLanguageModel (2-5x faster, 70% less VRAM)

Reward model (multi-objective):
  1. format_reward      — thinking chain present + answer in \\boxed{}
  2. correctness_reward — numerically / symbolically verifies math answers
  3. length_reward      — penalises both too-short and excessively long answers
  4. reasoning_reward   — rewards step-by-step structure in the thinking chain

Usage:
  python scripts/train_qwen3_grpo.py --config configs/qwen3_grpo.yaml
  python scripts/train_qwen3_grpo.py --model unsloth/Qwen3-8B --max-steps 100

Windows notes:
  - Flash Attention is NOT available natively; xformers is used instead.
  - Activate the isolated conda env before running:
      conda activate unsloth-qwen3-grpo
  - Load credentials from .env or export env vars manually.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# ─── Load .env early so HF_TOKEN / WANDB_API_KEY are available ───────────────
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("qwen3_grpo")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Model
    model: str = "unsloth/Qwen3-8B"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # GRPO
    output_dir: str = "outputs/qwen3-grpo"
    num_train_epochs: int = 1
    max_steps: int = -1               # -1 means use num_train_epochs
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    num_generations: int = 6          # Responses per prompt for GRPO
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    temperature: float = 0.9
    top_p: float = 0.95
    logging_steps: int = 1
    save_steps: int = 50
    eval_steps: int = 50
    fp16: bool = False
    bf16: bool = True                 # Ampere+ GPUs; set False + fp16=True for older
    gradient_checkpointing: str = "unsloth"  # Unsloth's memory-efficient variant

    # Dataset
    dataset: str = "openai/gsm8k"
    dataset_config: str = "main"
    dataset_split: str = "train"
    dataset_eval_split: str = "test"
    max_train_samples: int = -1       # -1 = use all
    max_eval_samples: int = 200

    # Logging
    report_to: str = "wandb"          # "wandb" | "tensorboard" | "none"
    run_name: str = ""
    push_to_hub: bool = False
    hub_model_id: str = ""

    # Reward weights (must sum to 1.0)
    weight_format: float = 0.3
    weight_correctness: float = 0.4
    weight_reasoning: float = 0.2
    weight_length: float = 0.1

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                log.warning("Unknown config key '%s' — ignored.", k)
        return cfg

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        cfg = cls()
        for k, v in vars(args).items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions
# ─────────────────────────────────────────────────────────────────────────────

_BOXED_RE    = re.compile(r"\\boxed\{(.+?)\}", re.DOTALL)
_THINKING_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_STEP_RE     = re.compile(r"(?:step\s*\d+|[①②③④⑤]|\d+\.\s|\n-\s)", re.IGNORECASE)


def _extract_boxed(text: str) -> str | None:
    """Return the last \\boxed{} content, or None."""
    hits = _BOXED_RE.findall(text)
    return hits[-1].strip() if hits else None


def _extract_thinking(text: str) -> str:
    """Return text inside <think>…</think>, empty string if absent."""
    m = _THINKING_RE.search(text)
    return m.group(1).strip() if m else ""


def _numbers_equal(a: str, b: str) -> bool:
    """Numerically compare two string representations of numbers."""
    try:
        return math.isclose(float(a.replace(",", "")), float(b.replace(",", "")), rel_tol=1e-4)
    except (ValueError, TypeError):
        return a.strip() == b.strip()


# --- 1. Format reward --------------------------------------------------------

def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Rewards the presence of:
      - A <think>…</think> block (Qwen3 extended thinking)
      - A \\boxed{} answer at the end
    """
    scores: list[float] = []
    for comp in completions:
        score = 0.0
        thinking = _extract_thinking(comp)
        if thinking:
            score += 0.5                    # Thinking block present
            if len(thinking.split()) >= 10:
                score += 0.2               # Non-trivial thinking
        if _extract_boxed(comp) is not None:
            score += 0.3                    # Answer formatted correctly
        scores.append(score)
    return scores


# --- 2. Correctness reward ---------------------------------------------------

def correctness_reward(
    completions: list[str],
    answers: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """
    Rewards correct numerical answers extracted from \\boxed{}.
    Falls back gracefully when ground-truth answers are not provided.
    """
    if not answers:
        return [0.0] * len(completions)

    scores: list[float] = []
    for comp, expected in zip(completions, answers):
        score = 0.0
        predicted = _extract_boxed(comp)
        if predicted is not None:
            if _numbers_equal(predicted, str(expected)):
                score = 1.0                 # Exact / numerically equal
            else:
                # Partial credit: answer appears somewhere in response
                if str(expected) in comp:
                    score = 0.3
        scores.append(score)
    return scores


# --- 3. Reasoning-chain reward -----------------------------------------------

def reasoning_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Rewards structured step-by-step reasoning inside <think>.
    Metrics: number of explicit steps, use of equations, logical connectives.
    """
    scores: list[float] = []
    for comp in completions:
        thinking = _extract_thinking(comp)
        if not thinking:
            scores.append(0.0)
            continue

        score = 0.0
        steps = len(_STEP_RE.findall(thinking))
        score += min(steps * 0.08, 0.4)    # Up to 0.4 for multi-step reasoning

        # Equations / calculations
        eq_count = thinking.count("=")
        score += min(eq_count * 0.03, 0.2)

        # Logical connectives
        connectives = sum(
            1 for w in ["therefore", "because", "since", "thus", "so", "hence"]
            if w in thinking.lower()
        )
        score += min(connectives * 0.05, 0.2)

        # Penalise copying the question verbatim (lazy thinking)
        if len(thinking.split()) < 5:
            score = 0.0

        scores.append(round(min(score, 1.0), 4))
    return scores


# --- 4. Length reward --------------------------------------------------------

def length_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Gaussian-shaped reward centred around an ideal completion length.
    Penalises both trivially short and unnecessarily long responses.
    """
    IDEAL   = 350     # ideal word count (thinking + answer combined)
    SIGMA   = 200     # tolerance band
    scores: list[float] = []
    for comp in completions:
        words = len(comp.split())
        score = math.exp(-0.5 * ((words - IDEAL) / SIGMA) ** 2)
        scores.append(round(score, 4))
    return scores


# --- Composite reward --------------------------------------------------------

def make_composite_reward(cfg: TrainingConfig):
    """Factory that returns a single reward function with configured weights."""
    w_fmt   = cfg.weight_format
    w_corr  = cfg.weight_correctness
    w_reas  = cfg.weight_reasoning
    w_len   = cfg.weight_length

    def composite_reward(completions: list[str], **kwargs) -> list[float]:
        fmt_scores  = format_reward(completions, **kwargs)
        corr_scores = correctness_reward(completions, **kwargs)
        reas_scores = reasoning_reward(completions, **kwargs)
        len_scores  = length_reward(completions, **kwargs)

        return [
            w_fmt  * f
            + w_corr * c
            + w_reas * r
            + w_len  * ln
            for f, c, r, ln in zip(fmt_scores, corr_scores, reas_scores, len_scores)
        ]

    return composite_reward


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful and rigorous reasoning assistant. "
    "Think step-by-step inside <think>...</think> before giving your final answer. "
    "Always present your final numerical answer inside \\boxed{}."
)


def _build_prompt(example: dict, tokenizer) -> dict:
    """Convert a raw GSM8K example into a Qwen3 chat prompt."""
    question = example.get("question", example.get("prompt", ""))
    answer   = example.get("answer",   example.get("solution", ""))

    # Extract the numeric part from GSM8K answers  (e.g. "#### 42" → "42")
    numeric_answer = ""
    if "####" in str(answer):
        numeric_answer = answer.split("####")[-1].strip().replace(",", "")

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,           # Qwen3: emit <think> token
    )
    return {"prompt": prompt_text, "answer": numeric_answer}


def load_and_prepare_dataset(cfg: TrainingConfig, tokenizer):
    from datasets import load_dataset

    log.info("Loading dataset: %s / %s", cfg.dataset, cfg.dataset_config or "default")
    raw_train = load_dataset(cfg.dataset, cfg.dataset_config or None, split=cfg.dataset_split)
    raw_eval  = load_dataset(cfg.dataset, cfg.dataset_config or None, split=cfg.dataset_eval_split)

    if cfg.max_train_samples > 0:
        raw_train = raw_train.select(range(min(cfg.max_train_samples, len(raw_train))))
    if cfg.max_eval_samples > 0:
        raw_eval = raw_eval.select(range(min(cfg.max_eval_samples, len(raw_eval))))

    fn_kwargs = {"tokenizer": tokenizer}
    train_ds = raw_train.map(_build_prompt, fn_kwargs=fn_kwargs, remove_columns=raw_train.column_names)
    eval_ds  = raw_eval.map(_build_prompt,  fn_kwargs=fn_kwargs, remove_columns=raw_eval.column_names)

    log.info("Train: %d examples | Eval: %d examples", len(train_ds), len(eval_ds))
    return train_ds, eval_ds


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainingConfig) -> None:
    log.info("=== Qwen3 Advanced GRPO Training ===")
    log.info("Model        : %s", cfg.model)
    log.info("Output dir   : %s", cfg.output_dir)
    log.info("Quantization : %s", "4-bit" if cfg.load_in_4bit else "8-bit" if cfg.load_in_8bit else "16-bit")

    # ── 1. Load model + tokenizer via Unsloth ────────────────────────────────
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = cfg.model,
        max_seq_length  = cfg.max_seq_length,
        load_in_4bit    = cfg.load_in_4bit,
        load_in_8bit    = cfg.load_in_8bit,
        dtype           = None,          # auto-detect bfloat16/float16
    )

    # ── 2. Attach LoRA adapters ───────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r                           = cfg.lora_rank,
        lora_alpha                  = cfg.lora_alpha,
        lora_dropout                = cfg.lora_dropout,
        target_modules              = cfg.target_modules,
        use_gradient_checkpointing  = cfg.gradient_checkpointing,
        random_state                = 42,
        use_rslora                  = False,
    )

    # ── 3. Enable Unsloth GRPO patch ─────────────────────────────────────────
    from unsloth.models.rl import PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)

    # ── 4. Dataset ────────────────────────────────────────────────────────────
    train_ds, eval_ds = load_and_prepare_dataset(cfg, tokenizer)

    # ── 5. Composite reward function ──────────────────────────────────────────
    reward_fn = make_composite_reward(cfg)

    log.info(
        "Reward weights — format:%.2f correctness:%.2f reasoning:%.2f length:%.2f",
        cfg.weight_format, cfg.weight_correctness, cfg.weight_reasoning, cfg.weight_length,
    )

    # ── 6. GRPOConfig ─────────────────────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    run_name = cfg.run_name or f"{Path(cfg.model).name}-grpo"
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        output_dir                  = cfg.output_dir,
        run_name                    = run_name,
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        learning_rate               = cfg.learning_rate,
        lr_scheduler_type           = cfg.lr_scheduler_type,
        warmup_ratio                = cfg.warmup_ratio,
        num_train_epochs            = cfg.num_train_epochs,
        max_steps                   = cfg.max_steps if cfg.max_steps > 0 else -1,
        fp16                        = cfg.fp16,
        bf16                        = cfg.bf16,
        num_generations             = cfg.num_generations,
        max_prompt_length           = cfg.max_prompt_length,
        max_completion_length       = cfg.max_completion_length,
        temperature                 = cfg.temperature,
        top_p                       = cfg.top_p,
        logging_steps               = cfg.logging_steps,
        save_steps                  = cfg.save_steps,
        eval_steps                  = cfg.eval_steps,
        evaluation_strategy         = "steps",
        report_to                   = cfg.report_to if cfg.report_to != "none" else [],
        push_to_hub                 = cfg.push_to_hub,
        hub_model_id                = cfg.hub_model_id or None,
        remove_unused_columns       = False,    # Required for custom reward kwargs
    )

    # ── 7. Build trainer ──────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        reward_funcs    = [reward_fn],
        args            = training_args,
    )

    # ── 8. Train ──────────────────────────────────────────────────────────────
    log.info("Starting GRPO training...")
    trainer.train()

    # ── 9. Save ───────────────────────────────────────────────────────────────
    log.info("Saving LoRA adapters to %s", cfg.output_dir)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Save effective config alongside the model
    cfg_out = Path(cfg.output_dir) / "training_config.json"
    with open(cfg_out, "w", encoding="utf-8") as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    log.info("Training complete. Artefacts saved to: %s", cfg.output_dir)

    # ── 10. Optional: push to Hub ─────────────────────────────────────────────
    if cfg.push_to_hub and cfg.hub_model_id:
        log.info("Pushing to HuggingFace Hub: %s", cfg.hub_model_id)
        model.push_to_hub(cfg.hub_model_id)
        tokenizer.push_to_hub(cfg.hub_model_id)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qwen3 Advanced GRPO Training — Unsloth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",   type=str, help="Path to YAML config file (overrides defaults)")
    p.add_argument("--model",    type=str, help="HuggingFace model ID or local path")
    p.add_argument("--dataset",  type=str, help="HuggingFace dataset name")
    p.add_argument("--output-dir", dest="output_dir", type=str, help="Where to save checkpoints")
    p.add_argument("--max-steps",  dest="max_steps",  type=int, help="Hard cap on training steps")
    p.add_argument("--num-epochs", dest="num_train_epochs", type=int, help="Training epochs")
    p.add_argument("--lora-rank",  dest="lora_rank",  type=int, help="LoRA rank")
    p.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=None,
                   help="4-bit quantization (default on)")
    p.add_argument("--bf16",     action="store_true", default=None, help="Use bfloat16")
    p.add_argument("--fp16",     action="store_true", default=None, help="Use float16")
    p.add_argument("--report-to", dest="report_to",  type=str, choices=["wandb","tensorboard","none"])
    p.add_argument("--run-name",   dest="run_name",   type=str)
    p.add_argument("--push-to-hub", dest="push_to_hub", action="store_true", default=None)
    p.add_argument("--hub-model-id", dest="hub_model_id", type=str)
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Layer config: defaults → YAML file → CLI args
    if args.config:
        cfg = TrainingConfig.from_yaml(args.config)
        # Apply CLI overrides on top of YAML
        for k, v in vars(args).items():
            if k == "config":
                continue
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        cfg = TrainingConfig.from_args(args)

    train(cfg)


if __name__ == "__main__":
    main()
