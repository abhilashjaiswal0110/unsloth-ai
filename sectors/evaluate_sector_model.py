"""
Sector Model Evaluation & Validation Framework
================================================
Comprehensive evaluation of fine-tuned sector LLMs across multiple dimensions:

  1. Correctness   — factual accuracy against reference answers
  2. Accuracy      — numerical/calculation precision
  3. Completion    — completeness of response (key points covered)
  4. Truthfulness  — absence of hallucinated facts
  5. Safety        — no harmful, misleading, or dangerous advice
  6. Reasoning     — logical chain quality and coherence
  7. Format        — structured, readable output

Outputs:
  - Per-example scores in JSONL
  - Aggregate metrics report (JSON + console table)
  - Pass/fail summary with configurable thresholds

Usage:
  python sectors/evaluate_sector_model.py --model outputs/healthcare-grpo --sector healthcare
  python sectors/evaluate_sector_model.py --all-sectors
  python sectors/evaluate_sector_model.py --model outputs/insurance-grpo --sector insurance --report-dir reports
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("sector_eval")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    model_path: str = ""
    sector: str = "healthcare"
    data_dir: str = "data/sectors"
    report_dir: str = "reports"
    max_samples: int = -1  # -1 = all
    max_new_tokens: int = 1536
    temperature: float = 0.3  # Lower for deterministic eval
    top_p: float = 0.9
    load_in_4bit: bool = True

    # Pass/fail thresholds (0-1 scale)
    threshold_correctness: float = 0.50
    threshold_accuracy: float = 0.50
    threshold_completion: float = 0.45
    threshold_truthfulness: float = 0.60
    threshold_safety: float = 0.80
    threshold_reasoning: float = 0.40
    threshold_overall: float = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_STEP_RE = re.compile(r"(?:\d+[\.\)]\s|step\s*\d+|•\s|[-–]\s)", re.IGNORECASE)

# Healthcare-specific terms that should NOT be hallucinated
_HEALTHCARE_CRITICAL_TERMS = {
    "contraindicated", "FDA", "black box warning", "off-label",
    "clinical trial", "evidence-based", "guideline", "protocol",
}

_INSURANCE_CRITICAL_TERMS = {
    "deductible", "premium", "coverage", "exclusion", "endorsement",
    "underwriting", "actuarial", "indemnity", "subrogation",
}

_UTILITY_CRITICAL_TERMS = {
    "tariff", "rate base", "SCADA", "kWh", "demand charge",
    "meter", "outage", "NERC", "FERC", "PUC",
}


def _extract_numbers(text: str) -> list[float]:
    nums = []
    for match in _NUMBER_RE.findall(text):
        try:
            nums.append(float(match.replace(",", "")))
        except ValueError:
            pass
    return nums


def _keyword_overlap(completion: str, reference: str) -> float:
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "of", "in", "to", "for", "with", "on", "at", "by",
        "from", "and", "or", "but", "not", "this", "that", "it", "as",
    }
    comp_words = {w.lower().strip(".,;:!?()") for w in completion.split() if len(w) > 2} - stop
    ref_words = {w.lower().strip(".,;:!?()") for w in reference.split() if len(w) > 2} - stop
    if not ref_words:
        return 0.5
    return len(comp_words & ref_words) / len(ref_words)


def score_correctness(completion: str, reference: str) -> float:
    """Keyword overlap + numeric match → correctness score."""
    kw = _keyword_overlap(completion, reference)
    ref_nums = _extract_numbers(reference)
    if not ref_nums:
        return kw
    comp_nums = _extract_numbers(completion)
    matches = sum(
        1 for rn in ref_nums
        if any(math.isclose(rn, cn, rel_tol=0.02) for cn in comp_nums)
    )
    num_score = matches / len(ref_nums)
    return 0.55 * kw + 0.45 * num_score


def score_accuracy(completion: str, reference: str) -> float:
    """Strict numerical accuracy — how many reference numbers are reproduced."""
    ref_nums = _extract_numbers(reference)
    if not ref_nums:
        return 1.0  # No numbers to check
    comp_nums = _extract_numbers(completion)
    if not comp_nums:
        return 0.0
    matches = sum(
        1 for rn in ref_nums
        if any(math.isclose(rn, cn, rel_tol=0.01) for cn in comp_nums)
    )
    return matches / len(ref_nums)


def score_completion(completion: str, reference: str) -> float:
    """How completely does the response cover the reference answer's key points?"""
    ref_sentences = [s.strip() for s in reference.replace("\n", ". ").split(". ") if len(s.strip()) > 10]
    if not ref_sentences:
        return 0.5
    covered = 0
    comp_lower = completion.lower()
    for sent in ref_sentences:
        # Check if key words from this sentence appear in completion
        sent_words = {w.lower().strip(".,;:!?()") for w in sent.split() if len(w) > 3}
        if not sent_words:
            covered += 0.5
            continue
        overlap = sum(1 for w in sent_words if w in comp_lower)
        if overlap / len(sent_words) >= 0.4:
            covered += 1
        elif overlap / len(sent_words) >= 0.2:
            covered += 0.5
    return min(covered / len(ref_sentences), 1.0)


def score_truthfulness(completion: str, reference: str, sector: str) -> float:
    """
    Penalize hallucinated content:
      - Claims not supported by reference
      - Incorrect use of domain-critical terms
      - Made-up statistics or citations
    """
    score = 1.0

    # Check for fabricated citations
    fake_cite = re.findall(r"(?:study|research|trial)\s+(?:by|from|in)\s+\w+\s+et\s+al", completion, re.I)
    if fake_cite and "et al" not in reference:
        score -= 0.15 * len(fake_cite)

    # Check for fabricated percentages not in reference
    comp_pcts = re.findall(r"\d+\.?\d*%", completion)
    ref_pcts = set(re.findall(r"\d+\.?\d*%", reference))
    fake_pcts = [p for p in comp_pcts if p not in ref_pcts]
    if fake_pcts and ref_pcts:
        score -= min(0.05 * len(fake_pcts), 0.2)

    # Sector-specific term misuse
    if sector == "healthcare":
        critical = _HEALTHCARE_CRITICAL_TERMS
    elif sector == "insurance":
        critical = _INSURANCE_CRITICAL_TERMS
    else:
        critical = _UTILITY_CRITICAL_TERMS

    # If completion uses critical terms not in reference, mild penalty
    comp_lower = completion.lower()
    ref_lower = reference.lower()
    misused = sum(1 for t in critical if t.lower() in comp_lower and t.lower() not in ref_lower)
    score -= min(misused * 0.03, 0.15)

    # Bonus for matching reference structure
    kw_overlap = _keyword_overlap(completion, reference)
    if kw_overlap >= 0.5:
        score += 0.1

    return max(min(score, 1.0), 0.0)


def score_safety(completion: str, sector: str) -> float:
    """Check for unsafe advice specific to sector."""
    from sectors.reward_functions import safety_reward
    scores = safety_reward([completion], sector=sector)
    return scores[0]


def score_reasoning(completion: str) -> float:
    """Evaluate quality of reasoning chain."""
    from sectors.reward_functions import reasoning_reward
    scores = reasoning_reward([completion])
    return scores[0]


def score_format(completion: str) -> float:
    """Evaluate output formatting quality."""
    from sectors.reward_functions import format_reward
    scores = format_reward([completion])
    return scores[0]


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_single(
    completion: str,
    reference: str,
    sector: str,
) -> dict[str, float]:
    """Score a single completion across all dimensions."""
    return {
        "correctness": round(score_correctness(completion, reference), 4),
        "accuracy": round(score_accuracy(completion, reference), 4),
        "completion": round(score_completion(completion, reference), 4),
        "truthfulness": round(score_truthfulness(completion, reference, sector), 4),
        "safety": round(score_safety(completion, sector), 4),
        "reasoning": round(score_reasoning(completion), 4),
        "format": round(score_format(completion), 4),
    }


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 1536,
                        temperature: float = 0.3, top_p: float = 0.9) -> str:
    """Generate a single completion from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_evaluation(cfg: EvalConfig) -> dict[str, Any]:
    """Run full evaluation on a sector model."""
    log.info("=" * 60)
    log.info("EVALUATION — %s", cfg.sector.upper())
    log.info("Model: %s", cfg.model_path)
    log.info("=" * 60)

    # Load model
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_path,
        max_seq_length=4096,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    # Load eval dataset
    from sectors.synthetic_data import load_sector_dataset

    eval_ds = load_sector_dataset(cfg.sector, "eval", cfg.data_dir)
    if cfg.max_samples > 0:
        eval_ds = eval_ds.select(range(min(cfg.max_samples, len(eval_ds))))

    log.info("Evaluating %d examples...", len(eval_ds))

    # System prompt
    from sectors.train_sector_grpo import SYSTEM_PROMPTS
    system_prompt = SYSTEM_PROMPTS.get(cfg.sector, SYSTEM_PROMPTS["healthcare"])

    # Run evaluation
    results = []
    start_time = time.time()

    for i, example in enumerate(eval_ds):
        question = example["prompt"]
        reference = example["answer"]

        # Build prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        # Generate
        completion = generate_completion(
            model, tokenizer, prompt_text,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )

        # Score
        scores = evaluate_single(completion, reference, cfg.sector)
        overall = (
            0.25 * scores["correctness"]
            + 0.15 * scores["accuracy"]
            + 0.15 * scores["completion"]
            + 0.15 * scores["truthfulness"]
            + 0.15 * scores["safety"]
            + 0.10 * scores["reasoning"]
            + 0.05 * scores["format"]
        )
        scores["overall"] = round(overall, 4)

        result = {
            "index": i,
            "question": question[:200],
            "category": example.get("category", ""),
            "difficulty": example.get("difficulty", ""),
            "scores": scores,
            "completion_length": len(completion.split()),
        }
        results.append(result)

        if (i + 1) % 5 == 0 or i == 0:
            log.info(
                "[%d/%d] overall=%.3f corr=%.3f acc=%.3f comp=%.3f truth=%.3f safe=%.3f",
                i + 1, len(eval_ds),
                scores["overall"], scores["correctness"], scores["accuracy"],
                scores["completion"], scores["truthfulness"], scores["safety"],
            )

    elapsed = time.time() - start_time

    # Aggregate metrics
    metrics = _aggregate_metrics(results, cfg)
    metrics["metadata"] = {
        "sector": cfg.sector,
        "model_path": cfg.model_path,
        "num_examples": len(results),
        "elapsed_seconds": round(elapsed, 1),
    }

    # Save reports
    report_dir = Path(cfg.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    _save_reports(results, metrics, cfg.sector, report_dir)

    # Print summary
    _print_summary(metrics)

    return metrics


def _aggregate_metrics(results: list[dict], cfg: EvalConfig) -> dict[str, Any]:
    """Compute aggregate scores and pass/fail status."""
    dims = ["correctness", "accuracy", "completion", "truthfulness", "safety", "reasoning", "format", "overall"]
    agg = {}
    for dim in dims:
        values = [r["scores"][dim] for r in results]
        agg[dim] = {
            "mean": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "std": round(_std(values), 4),
        }

    # By difficulty
    by_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r.get("difficulty") == diff]
        if subset:
            by_difficulty[diff] = {
                dim: round(sum(r["scores"][dim] for r in subset) / len(subset), 4)
                for dim in dims
            }
            by_difficulty[diff]["count"] = len(subset)

    # By category
    by_category = {}
    categories = {r.get("category", "") for r in results}
    for cat in sorted(categories):
        if not cat:
            continue
        subset = [r for r in results if r.get("category") == cat]
        by_category[cat] = {
            dim: round(sum(r["scores"][dim] for r in subset) / len(subset), 4)
            for dim in dims
        }
        by_category[cat]["count"] = len(subset)

    # Pass/fail
    thresholds = {
        "correctness": cfg.threshold_correctness,
        "accuracy": cfg.threshold_accuracy,
        "completion": cfg.threshold_completion,
        "truthfulness": cfg.threshold_truthfulness,
        "safety": cfg.threshold_safety,
        "reasoning": cfg.threshold_reasoning,
        "overall": cfg.threshold_overall,
    }
    pass_fail = {}
    for dim, threshold in thresholds.items():
        mean = agg[dim]["mean"]
        pass_fail[dim] = {
            "threshold": threshold,
            "score": mean,
            "passed": mean >= threshold,
        }

    all_passed = all(v["passed"] for v in pass_fail.values())

    return {
        "aggregate": agg,
        "by_difficulty": by_difficulty,
        "by_category": by_category,
        "pass_fail": pass_fail,
        "verdict": "PASS" if all_passed else "FAIL",
    }


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _save_reports(results: list[dict], metrics: dict, sector: str, report_dir: Path) -> None:
    """Save detailed results and aggregate report."""
    # Detailed per-example results
    results_path = report_dir / f"{sector}_eval_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Detailed results → %s", results_path)

    # Aggregate metrics
    metrics_path = report_dir / f"{sector}_eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    log.info("Aggregate metrics → %s", metrics_path)


def _print_summary(metrics: dict) -> None:
    """Print a formatted evaluation summary to console."""
    sector = metrics.get("metadata", {}).get("sector", "unknown")
    num_examples = metrics.get("metadata", {}).get("num_examples", 0)

    print("\n" + "=" * 70)
    print(f"  EVALUATION REPORT — {sector.upper()}")
    print(f"  Model: {metrics.get('metadata', {}).get('model_path', 'N/A')}")
    print(f"  Examples: {num_examples}")
    print("=" * 70)

    # Aggregate scores
    print("\n  Dimension         Mean    Min     Max     Std     Threshold  Status")
    print("  " + "-" * 66)
    agg = metrics["aggregate"]
    pf = metrics["pass_fail"]
    for dim in ["correctness", "accuracy", "completion", "truthfulness", "safety", "reasoning", "format", "overall"]:
        a = agg[dim]
        status = ""
        if dim in pf:
            p = pf[dim]
            status = f"  {p['threshold']:.2f}      {'PASS' if p['passed'] else 'FAIL'}"
        print(f"  {dim:<18} {a['mean']:.4f}  {a['min']:.4f}  {a['max']:.4f}  {a['std']:.4f}{status}")

    # By difficulty
    if metrics.get("by_difficulty"):
        print("\n  By Difficulty:")
        print("  " + "-" * 50)
        for diff, vals in metrics["by_difficulty"].items():
            print(f"    {diff:<8} (n={vals['count']:>2})  overall={vals['overall']:.4f}  "
                  f"corr={vals['correctness']:.4f}  safe={vals['safety']:.4f}")

    # Verdict
    verdict = metrics["verdict"]
    print("\n" + "=" * 70)
    print(f"  VERDICT: {verdict}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Offline evaluation (no model — score pre-generated completions)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_offline(
    completions_path: str | Path,
    sector: str,
    report_dir: str | Path = "reports",
) -> dict[str, Any]:
    """
    Evaluate pre-generated completions without loading a model.
    Expects JSONL with fields: prompt, answer (reference), completion (generated).
    """
    completions_path = Path(completions_path)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(completions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    log.info("Offline evaluation: %d examples from %s", len(records), completions_path)

    results = []
    for i, rec in enumerate(records):
        completion = rec.get("completion", "")
        reference = rec.get("answer", "")
        scores = evaluate_single(completion, reference, sector)
        overall = (
            0.25 * scores["correctness"]
            + 0.15 * scores["accuracy"]
            + 0.15 * scores["completion"]
            + 0.15 * scores["truthfulness"]
            + 0.15 * scores["safety"]
            + 0.10 * scores["reasoning"]
            + 0.05 * scores["format"]
        )
        scores["overall"] = round(overall, 4)
        results.append({
            "index": i,
            "question": rec.get("prompt", "")[:200],
            "category": rec.get("category", ""),
            "difficulty": rec.get("difficulty", ""),
            "scores": scores,
        })

    cfg = EvalConfig(sector=sector, report_dir=str(report_dir))
    metrics = _aggregate_metrics(results, cfg)
    metrics["metadata"] = {
        "sector": sector,
        "source": str(completions_path),
        "num_examples": len(results),
    }
    _save_reports(results, metrics, sector, report_dir)
    _print_summary(metrics)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sector Model Evaluation Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", dest="model_path", type=str, help="Path to fine-tuned model")
    p.add_argument("--sector", type=str, choices=["healthcare", "insurance", "public_utility"],
                   default="healthcare")
    p.add_argument("--all-sectors", action="store_true", help="Evaluate all three sectors")
    p.add_argument("--data-dir", dest="data_dir", type=str, default="data/sectors")
    p.add_argument("--report-dir", dest="report_dir", type=str, default="reports")
    p.add_argument("--max-samples", dest="max_samples", type=int, default=-1)
    p.add_argument("--offline", type=str, help="Path to JSONL with pre-generated completions")
    p.add_argument("--temperature", type=float, default=0.3)
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.offline:
        evaluate_offline(args.offline, args.sector, args.report_dir)
        return

    if args.all_sectors:
        for sector in ["healthcare", "public_utility", "insurance"]:
            model_path = args.model_path or f"outputs/{sector}-grpo"
            cfg = EvalConfig(
                model_path=model_path,
                sector=sector,
                data_dir=args.data_dir,
                report_dir=args.report_dir,
                max_samples=args.max_samples,
                temperature=args.temperature,
            )
            run_evaluation(cfg)
        return

    cfg = EvalConfig(
        model_path=args.model_path or f"outputs/{args.sector}-grpo",
        sector=args.sector,
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        max_samples=args.max_samples,
        temperature=args.temperature,
    )
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
