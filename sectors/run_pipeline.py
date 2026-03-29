"""
End-to-End Sector GRPO Pipeline Orchestrator
=============================================
Single entry point that runs the complete pipeline for all sectors:

  1. Generate synthetic datasets
  2. Train GRPO models per sector
  3. Evaluate and validate each model
  4. Produce consolidated report

Usage:
  python sectors/run_pipeline.py                        # All sectors, full pipeline
  python sectors/run_pipeline.py --sector healthcare    # Single sector
  python sectors/run_pipeline.py --skip-train            # Only data gen + eval (uses existing models)
  python sectors/run_pipeline.py --skip-eval             # Only data gen + training
  python sectors/run_pipeline.py --max-steps 50         # Quick test run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("pipeline")

ALL_SECTORS = ["healthcare", "public_utility", "insurance"]


def step_generate_data(sectors: list[str], data_dir: str = "data/sectors") -> None:
    """Step 1: Generate synthetic datasets for all requested sectors."""
    log.info("━" * 50)
    log.info("STEP 1 — Generating Synthetic Datasets")
    log.info("━" * 50)

    from sectors.synthetic_data import generate_sector_dataset

    for sector in sectors:
        log.info("Generating: %s", sector)
        generate_sector_dataset(sector, data_dir)

    # Print dataset stats
    data_path = Path(data_dir)
    for sector in sectors:
        train = data_path / f"{sector}_train.jsonl"
        eval_ = data_path / f"{sector}_eval.jsonl"
        t_count = 0
        if train.exists():
            with open(train, encoding="utf-8") as f:
                t_count = sum(1 for _ in f)
        e_count = 0
        if eval_.exists():
            with open(eval_, encoding="utf-8") as f:
                e_count = sum(1 for _ in f)
        log.info("  %s — train: %d, eval: %d", sector, t_count, e_count)


def step_train(
    sectors: list[str],
    model: str | None = None,
    max_steps: int | None = None,
    num_epochs: int | None = None,
) -> dict[str, str]:
    """Step 2: Train GRPO models for each sector."""
    log.info("━" * 50)
    log.info("STEP 2 — GRPO Training")
    log.info("━" * 50)

    from sectors.train_sector_grpo import SectorTrainingConfig, train_sector

    output_dirs = {}
    configs_dir = _PROJECT_ROOT / "configs"

    for sector in sectors:
        cfg_path = configs_dir / f"{sector}_grpo.yaml"
        if cfg_path.exists():
            cfg = SectorTrainingConfig.from_yaml(cfg_path)
        else:
            cfg = SectorTrainingConfig(
                sector=sector, output_dir=f"outputs/{sector}-grpo"
            )

        if model:
            cfg.model = model
        if max_steps is not None:
            cfg.max_steps = max_steps
        if num_epochs is not None:
            cfg.num_train_epochs = num_epochs

        output_path = train_sector(cfg)
        output_dirs[sector] = str(output_path)

    return output_dirs


def step_evaluate(
    sectors: list[str],
    model_dirs: dict[str, str] | None = None,
    data_dir: str = "data/sectors",
    report_dir: str = "reports",
    max_samples: int = -1,
) -> dict[str, dict]:
    """Step 3: Evaluate each sector model."""
    log.info("━" * 50)
    log.info("STEP 3 — Evaluation & Validation")
    log.info("━" * 50)

    from sectors.evaluate_sector_model import EvalConfig, run_evaluation

    all_metrics = {}
    for sector in sectors:
        model_path = (model_dirs or {}).get(sector, f"outputs/{sector}-grpo")
        cfg = EvalConfig(
            model_path=model_path,
            sector=sector,
            data_dir=data_dir,
            report_dir=report_dir,
            max_samples=max_samples,
        )
        metrics = run_evaluation(cfg)
        all_metrics[sector] = metrics

    return all_metrics


def step_consolidated_report(
    all_metrics: dict[str, dict],
    report_dir: str = "reports",
) -> None:
    """Step 4: Write consolidated cross-sector report."""
    log.info("━" * 50)
    log.info("STEP 4 — Consolidated Report")
    log.info("━" * 50)

    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    # JSON report
    consolidated = {
        "sectors": {},
        "summary": {},
    }

    dims = [
        "correctness",
        "accuracy",
        "completion",
        "truthfulness",
        "safety",
        "reasoning",
        "format",
        "overall",
    ]

    for sector, metrics in all_metrics.items():
        agg = metrics.get("aggregate", {})
        consolidated["sectors"][sector] = {
            "verdict": metrics.get("verdict", "N/A"),
            "num_examples": metrics.get("metadata", {}).get("num_examples", 0),
            "scores": {dim: agg.get(dim, {}).get("mean", 0) for dim in dims},
        }

    # Cross-sector averages
    if consolidated["sectors"]:
        for dim in dims:
            values = [s["scores"].get(dim, 0) for s in consolidated["sectors"].values()]
            consolidated["summary"][dim] = round(sum(values) / len(values), 4)

    all_passed = all(
        s.get("verdict") == "PASS" for s in consolidated["sectors"].values()
    )
    consolidated["summary"]["overall_verdict"] = (
        "ALL PASS" if all_passed else "SOME FAIL"
    )

    out_path = report_path / "consolidated_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)
    log.info("Consolidated report → %s", out_path)

    # Console summary
    print("\n" + "=" * 70)
    print("  CONSOLIDATED PIPELINE REPORT")
    print("=" * 70)
    print(
        f"\n  {'Sector':<18} {'Verdict':<8} {'Overall':<8} {'Correct':<8} {'Accuracy':<8} {'Safety':<8}"
    )
    print("  " + "-" * 60)
    for sector, data in consolidated["sectors"].items():
        s = data["scores"]
        print(
            f"  {sector:<18} {data['verdict']:<8} {s['overall']:<8.4f} "
            f"{s['correctness']:<8.4f} {s['accuracy']:<8.4f} {s['safety']:<8.4f}"
        )
    print(f"\n  Overall verdict: {consolidated['summary']['overall_verdict']}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    sectors: list[str] | None = None,
    model: str | None = None,
    max_steps: int | None = None,
    num_epochs: int | None = None,
    skip_train: bool = False,
    skip_eval: bool = False,
    data_dir: str = "data/sectors",
    report_dir: str = "reports",
    max_eval_samples: int = -1,
) -> dict:
    """Run the complete pipeline."""
    sectors = sectors or ALL_SECTORS
    start = time.time()

    log.info("╔" + "═" * 58 + "╗")
    log.info("║   SECTOR GRPO PIPELINE — %s", ", ".join(s.upper() for s in sectors))
    log.info("╚" + "═" * 58 + "╝")

    # Step 1: Data generation
    step_generate_data(sectors, data_dir)

    # Step 2: Training
    model_dirs = {}
    if not skip_train:
        model_dirs = step_train(
            sectors, model=model, max_steps=max_steps, num_epochs=num_epochs
        )
    else:
        log.info("Skipping training (--skip-train)")
        model_dirs = {s: f"outputs/{s}-grpo" for s in sectors}

    # Step 3: Evaluation
    all_metrics = {}
    if not skip_eval:
        all_metrics = step_evaluate(
            sectors, model_dirs, data_dir, report_dir, max_eval_samples
        )
    else:
        log.info("Skipping evaluation (--skip-eval)")

    # Step 4: Consolidated report
    if all_metrics:
        step_consolidated_report(all_metrics, report_dir)

    elapsed = time.time() - start
    log.info("Pipeline completed in %.1f seconds", elapsed)

    return {"model_dirs": model_dirs, "metrics": all_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="End-to-End Sector GRPO Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sector", type=str, choices=ALL_SECTORS, help="Single sector (default: all)"
    )
    p.add_argument("--model", type=str, help="Base model override")
    p.add_argument("--max-steps", dest="max_steps", type=int, help="Max training steps")
    p.add_argument("--num-epochs", dest="num_epochs", type=int, help="Training epochs")
    p.add_argument("--skip-train", action="store_true", help="Skip training step")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    p.add_argument("--data-dir", dest="data_dir", type=str, default="data/sectors")
    p.add_argument("--report-dir", dest="report_dir", type=str, default="reports")
    p.add_argument("--max-eval-samples", dest="max_eval_samples", type=int, default=-1)

    args = p.parse_args()

    sectors = [args.sector] if args.sector else None

    run_pipeline(
        sectors=sectors,
        model=args.model,
        max_steps=args.max_steps,
        num_epochs=args.num_epochs,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    main()
