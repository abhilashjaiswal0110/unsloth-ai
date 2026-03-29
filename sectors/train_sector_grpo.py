"""
Sector-Specific GRPO Training Script
=====================================
RLHF + GRPO fine-tuning for domain models: healthcare, insurance, public utility.

Uses:
  - Synthetic sector data from sectors/synthetic_data.py
  - Multi-objective reward functions from sectors/reward_functions.py
  - Unsloth FastLanguageModel for efficient LoRA training
  - Qwen3 thinking mode (<think>…</think>) for reasoning chains

Usage:
  python sectors/train_sector_grpo.py --config configs/healthcare_grpo.yaml
  python sectors/train_sector_grpo.py --config configs/insurance_grpo.yaml
  python sectors/train_sector_grpo.py --config configs/public_utility_grpo.yaml
  python sectors/train_sector_grpo.py --sector healthcare --max-steps 50

All three sectors at once:
  python sectors/train_sector_grpo.py --all-sectors
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env
_env_file = _PROJECT_ROOT / ".env"
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
log = logging.getLogger("sector_grpo")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SectorTrainingConfig:
    # Model
    model: str = "unsloth/Qwen3-8B"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # GRPO Trainer
    output_dir: str = "outputs/sector-grpo"
    num_train_epochs: int = 3
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    num_generations: int = 6
    max_prompt_length: int = 768
    max_completion_length: int = 1536
    temperature: float = 0.7
    top_p: float = 0.95
    logging_steps: int = 1
    save_steps: int = 25
    eval_steps: int = 25
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: str = "unsloth"

    # Sector dataset
    sector: str = "healthcare"
    data_dir: str = "data/sectors"
    dataset_split: str = "train"
    dataset_eval_split: str = "eval"

    # Reward weights
    weight_format: float = 0.15
    weight_correctness: float = 0.35
    weight_reasoning: float = 0.25
    weight_completeness: float = 0.15
    weight_safety: float = 0.10

    # Logging
    report_to: str = "none"
    run_name: str = ""
    push_to_hub: bool = False
    hub_model_id: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SectorTrainingConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                log.warning("Unknown config key '%s' — ignored.", k)
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# System prompts per sector
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "healthcare": (
        "You are an expert medical assistant with deep knowledge of clinical medicine, "
        "pharmacology, diagnostics, and patient care. Think through each question carefully "
        "inside <think>...</think> before providing your answer. Always include evidence-based "
        "reasoning and cite relevant medical guidelines when applicable. If the question involves "
        "patient safety, err on the side of caution and recommend professional consultation."
    ),
    "public_utility": (
        "You are an expert utility operations advisor with deep knowledge of electricity, "
        "water, natural gas, and telecommunications infrastructure. Think through each question "
        "carefully inside <think>...</think> before providing your answer. For calculations, "
        "show all work. For operational questions, reference industry standards and regulatory "
        "requirements where applicable."
    ),
    "insurance": (
        "You are an expert insurance professional with deep knowledge of property & casualty, "
        "life, health, and commercial insurance. Think through each question carefully "
        "inside <think>...</think> before providing your answer. For calculations, show all "
        "steps. Reference relevant policy provisions, regulatory requirements, and industry "
        "standards. Always advise consulting a licensed professional for binding decisions."
    ),
}


def _build_sector_prompt(example: dict, tokenizer, sector: str) -> dict:
    """Convert a sector QA example into a chat-formatted prompt."""
    system_prompt = SYSTEM_PROMPTS.get(sector, SYSTEM_PROMPTS["healthcare"])
    question = example.get("prompt", "")
    answer = example.get("answer", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        # Fallback if tokenizer doesn't support enable_thinking
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return {"prompt": prompt_text, "answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────


def load_sector_data(cfg: SectorTrainingConfig, tokenizer):
    """Load sector data from JSONL and prepare for GRPO training."""
    from sectors.synthetic_data import generate_sector_dataset, load_sector_dataset

    data_dir = Path(cfg.data_dir)
    train_path = data_dir / f"{cfg.sector}_train.jsonl"
    eval_path = data_dir / f"{cfg.sector}_eval.jsonl"

    # Generate if either split does not exist
    if not train_path.exists() or not eval_path.exists():
        log.info("Generating synthetic data for sector: %s", cfg.sector)
        generate_sector_dataset(cfg.sector, data_dir)

    try:
        train_ds = load_sector_dataset(cfg.sector, "train", data_dir)
        eval_ds = load_sector_dataset(cfg.sector, "eval", data_dir)
    except Exception as exc:
        # Handle potential corruption or partial generation by regenerating both splits
        log.warning(
            "Failed to load sector datasets for %s (%s). Regenerating synthetic data.",
            cfg.sector,
            exc,
        )
        generate_sector_dataset(cfg.sector, data_dir)
        train_ds = load_sector_dataset(cfg.sector, "train", data_dir)
        eval_ds = load_sector_dataset(cfg.sector, "eval", data_dir)

    log.info("Loaded %s — train: %d, eval: %d", cfg.sector, len(train_ds), len(eval_ds))

    # Apply chat template formatting
    fn_kwargs = {"tokenizer": tokenizer, "sector": cfg.sector}
    train_ds = train_ds.map(
        _build_sector_prompt,
        fn_kwargs=fn_kwargs,
        remove_columns=[
            c for c in train_ds.column_names if c not in ("prompt", "answer")
        ],
    )
    eval_ds = eval_ds.map(
        _build_sector_prompt,
        fn_kwargs=fn_kwargs,
        remove_columns=[
            c for c in eval_ds.column_names if c not in ("prompt", "answer")
        ],
    )

    return train_ds, eval_ds


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def train_sector(cfg: SectorTrainingConfig) -> Path:
    """Train a single sector model with GRPO."""
    from sectors.reward_functions import make_sector_reward

    log.info("=" * 60)
    log.info("Sector GRPO Training — %s", cfg.sector.upper())
    log.info("=" * 60)
    log.info("Model        : %s", cfg.model)
    log.info("Output dir   : %s", cfg.output_dir)
    log.info(
        "Quantization : %s",
        "4-bit" if cfg.load_in_4bit else "8-bit" if cfg.load_in_8bit else "16-bit",
    )

    # 1. Load model + tokenizer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        dtype=None,
    )

    # 2. Attach LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
        random_state=42,
        use_rslora=False,
    )

    # 3. Enable GRPO patch
    from unsloth.models.rl import PatchFastRL

    PatchFastRL("GRPO", FastLanguageModel)

    # 4. Load sector dataset
    train_ds, eval_ds = load_sector_data(cfg, tokenizer)

    # 5. Build sector-specific composite reward
    reward_fn = make_sector_reward(
        sector=cfg.sector,
        weight_format=cfg.weight_format,
        weight_correctness=cfg.weight_correctness,
        weight_reasoning=cfg.weight_reasoning,
        weight_completeness=cfg.weight_completeness,
        weight_safety=cfg.weight_safety,
    )

    log.info(
        "Reward weights — fmt:%.2f corr:%.2f reas:%.2f comp:%.2f safe:%.2f",
        cfg.weight_format,
        cfg.weight_correctness,
        cfg.weight_reasoning,
        cfg.weight_completeness,
        cfg.weight_safety,
    )

    # 6. GRPO config
    from trl import GRPOConfig, GRPOTrainer

    run_name = cfg.run_name or f"{cfg.sector}-qwen3-grpo"
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect bf16 support if neither fp16 nor bf16 is explicitly set
    import torch

    use_bf16 = cfg.bf16
    use_fp16 = cfg.fp16
    if not use_bf16 and not use_fp16:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16
        log.info("Auto-detected precision: %s", "bf16" if use_bf16 else "fp16")

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps if cfg.max_steps > 0 else -1,
        fp16=use_fp16,
        bf16=use_bf16,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps",
        report_to=cfg.report_to if cfg.report_to != "none" else [],
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id or None,
        remove_unused_columns=False,
    )

    # 7. Build trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=[reward_fn],
        args=training_args,
    )

    # 8. Train
    log.info("Starting GRPO training for %s...", cfg.sector)
    trainer.train()

    # 9. Save
    log.info("Saving model to %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training config
    cfg_out = output_dir / "sector_training_config.json"
    with open(cfg_out, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in vars(cfg).items()}, f, indent=2, default=str)

    log.info("✓ %s training complete → %s", cfg.sector, output_dir)

    # 10. Optional hub push
    if cfg.push_to_hub and cfg.hub_model_id:
        log.info("Pushing to Hub: %s", cfg.hub_model_id)
        model.push_to_hub(cfg.hub_model_id)
        tokenizer.push_to_hub(cfg.hub_model_id)

    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sector-Specific GRPO Training — Healthcare / Insurance / Public Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, help="Path to YAML config file")
    p.add_argument(
        "--sector",
        type=str,
        choices=["healthcare", "insurance", "public_utility"],
        help="Sector to train (overrides config)",
    )
    p.add_argument(
        "--all-sectors",
        action="store_true",
        help="Train all three sectors sequentially",
    )
    p.add_argument("--model", type=str, help="HuggingFace model ID")
    p.add_argument("--output-dir", dest="output_dir", type=str, help="Output directory")
    p.add_argument("--max-steps", dest="max_steps", type=int, help="Max training steps")
    p.add_argument(
        "--num-epochs", dest="num_train_epochs", type=int, help="Training epochs"
    )
    p.add_argument("--lora-rank", dest="lora_rank", type=int, help="LoRA rank")
    p.add_argument(
        "--report-to",
        dest="report_to",
        type=str,
        choices=["wandb", "tensorboard", "none"],
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.all_sectors:
        configs_dir = _PROJECT_ROOT / "configs"
        for sector in ["healthcare", "public_utility", "insurance"]:
            cfg_path = configs_dir / f"{sector}_grpo.yaml"
            if cfg_path.exists():
                cfg = SectorTrainingConfig.from_yaml(cfg_path)
            else:
                cfg = SectorTrainingConfig(
                    sector=sector, output_dir=f"outputs/{sector}-grpo"
                )
            # Apply CLI overrides
            if args.max_steps is not None:
                cfg.max_steps = args.max_steps
            if args.model is not None:
                cfg.model = args.model
            train_sector(cfg)
        return

    # Single sector
    if args.config:
        cfg = SectorTrainingConfig.from_yaml(args.config)
    else:
        cfg = SectorTrainingConfig()

    # CLI overrides
    for k, v in vars(args).items():
        if k in ("config", "all_sectors"):
            continue
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    train_sector(cfg)


if __name__ == "__main__":
    main()
