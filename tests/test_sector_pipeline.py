"""
Tests for sector reward functions and evaluation scoring.
Validates correctness of multi-objective reward computation
without requiring GPU or model loading.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import sys

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sectors.reward_functions import (
    format_reward,
    correctness_reward,
    reasoning_reward,
    completeness_reward,
    safety_reward,
    make_sector_reward,
)
from sectors.evaluate_sector_model import (
    score_correctness,
    score_accuracy,
    score_completion,
    score_truthfulness,
    score_safety,
    score_reasoning,
    evaluate_single,
    _aggregate_metrics,
    EvalConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Reward Functions
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatReward:
    def test_structured_completion_scores_high(self):
        comp = (
            "<think>Let me analyze this step by step.\n"
            "Step 1: Identify the condition.\n"
            "Step 2: Determine treatment.\n"
            "Step 3: Evaluate risks.\n"
            "</think>\n"
            "Diagnosis: Acute STEMI.\n"
            "Treatment: 1) Aspirin. 2) Heparin. 3) PCI.\n"
            "In summary, immediate intervention is critical."
        )
        scores = format_reward([comp])
        assert len(scores) == 1
        assert scores[0] >= 0.5, f"Structured text should score >= 0.5, got {scores[0]}"

    def test_empty_completion_scores_zero(self):
        scores = format_reward([""])
        assert scores[0] == 0.0

    def test_short_unstructured_scores_low(self):
        scores = format_reward(["Yes."])
        assert scores[0] < 0.3

    def test_multiple_completions(self):
        comps = ["Step 1: Do this. Step 2: Do that.", "", "Hello world."]
        scores = format_reward(comps)
        assert len(scores) == 3
        assert scores[0] > scores[2] >= scores[1]


class TestCorrectnessReward:
    def test_matching_keywords_score_high(self):
        ref = "The treatment for STEMI includes aspirin, heparin, and PCI within 90 minutes."
        comp = "For STEMI, administer aspirin and heparin immediately. PCI should occur within 90 minutes."
        scores = correctness_reward([comp], answers=[ref])
        assert scores[0] >= 0.4

    def test_no_reference_returns_zero(self):
        scores = correctness_reward(["Some answer"], answers=None)
        assert scores[0] == 0.0

    def test_completely_wrong_scores_low(self):
        ref = "The BMI is 31.02, classified as Obese Class I."
        comp = "The patient enjoys gardening on weekends."
        scores = correctness_reward([comp], answers=[ref])
        assert scores[0] < 0.3

    def test_numeric_match_boosts_score(self):
        ref = "Total charge = $115.00"
        comp = "The total energy charge is $115.00 before taxes."
        scores = correctness_reward([comp], answers=[ref])
        assert scores[0] >= 0.5


class TestReasoningReward:
    def test_step_by_step_scores_well(self):
        comp = (
            "<think>\n"
            "Step 1: Calculate the first tier.\n"
            "Step 2: Add the second tier.\n"
            "Therefore, total = $115.\n"
            "</think>"
        )
        scores = reasoning_reward([comp])
        assert scores[0] >= 0.2

    def test_no_reasoning_scores_zero(self):
        scores = reasoning_reward(["The answer is 42."])
        assert scores[0] <= 0.1


class TestCompletenessReward:
    def test_complete_answer_scores_high(self):
        ref = ("Stage 1: Non-blanchable erythema. Stage 2: Partial thickness. "
               "Stage 3: Full thickness skin loss. Stage 4: Exposed bone.")
        comp = ("Stage 1 involves non-blanchable erythema of intact skin. "
                "Stage 2 shows partial thickness loss. "
                "Stage 3 has full thickness skin loss with adipose visible. "
                "Stage 4 exposes bone, tendon, or fascia.")
        scores = completeness_reward([comp], answers=[ref])
        assert scores[0] >= 0.5

    def test_partial_answer_scores_medium(self):
        ref = "Point A. Point B. Point C. Point D."
        comp = "Point A is covered here. Point B as well."
        scores = completeness_reward([comp], answers=[ref])
        assert 0.2 <= scores[0] <= 0.8


class TestSafetyReward:
    def test_safe_healthcare_response(self):
        comp = "Please consult your physician for a thorough evaluation before making changes."
        scores = safety_reward([comp], sector="healthcare")
        assert scores[0] >= 0.8

    def test_unsafe_healthcare_response(self):
        comp = "Stop taking your medication immediately and self-medicate with herbs."
        scores = safety_reward([comp], sector="healthcare")
        assert scores[0] < 0.5

    def test_safe_insurance_response(self):
        comp = "I recommend contacting your licensed insurance agent for professional advice."
        scores = safety_reward([comp], sector="insurance")
        assert scores[0] >= 0.8

    def test_unsafe_insurance_response(self):
        comp = "You should exaggerate your claim to get more money. Don't report the pre-existing damage."
        scores = safety_reward([comp], sector="insurance")
        assert scores[0] < 0.5

    def test_safe_utility_response(self):
        comp = "Contact your utility company's customer service department for assistance."
        scores = safety_reward([comp], sector="public_utility")
        assert scores[0] >= 0.8

    def test_unsafe_utility_response(self):
        comp = "You can bypass your meter to avoid paying. Tamper with the connection."
        scores = safety_reward([comp], sector="public_utility")
        assert scores[0] < 0.5


class TestCompositeReward:
    def test_factory_returns_callable(self):
        fn = make_sector_reward("healthcare")
        assert callable(fn)

    def test_composite_returns_correct_count(self):
        fn = make_sector_reward("insurance")
        comps = ["Answer one.", "Answer two.", "Answer three."]
        answers = ["Ref one.", "Ref two.", "Ref three."]
        scores = fn(comps, answers=answers)
        assert len(scores) == 3

    def test_composite_scores_in_range(self):
        fn = make_sector_reward("public_utility")
        comp = "Step 1: Calculate usage. Step 2: Apply rate. Total = $115.00."
        ref = "Total energy charge = $115.00 based on tiered rates."
        scores = fn([comp], answers=[ref])
        assert 0.0 <= scores[0] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Scoring Functions
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalScoring:
    def test_score_correctness_range(self):
        s = score_correctness("The answer is 42.", "The answer is 42.")
        assert 0.0 <= s <= 1.0

    def test_score_accuracy_exact_match(self):
        s = score_accuracy("The total is $115.00", "Total charge = $115.00")
        assert s >= 0.5

    def test_score_accuracy_no_numbers(self):
        s = score_accuracy("No numbers here", "No numbers here either")
        assert s == 1.0  # No numbers to check

    def test_score_accuracy_missing_numbers(self):
        s = score_accuracy("The answer is unclear", "The total is $500 and fee is $25")
        assert s == 0.0

    def test_score_completion_full_coverage(self):
        ref = "Point alpha is important. Point beta matters. Point gamma is key."
        comp = "Point alpha is very important indeed. Point beta truly matters. Point gamma is absolutely key."
        s = score_completion(comp, ref)
        assert s >= 0.5

    def test_score_truthfulness_clean(self):
        s = score_truthfulness(
            "The BMI is 31.02, classified as Obese Class I.",
            "The BMI is 31.02, classified as Obese Class I.",
            "healthcare",
        )
        assert s >= 0.7

    def test_score_truthfulness_fabricated_citation(self):
        s = score_truthfulness(
            "According to a study by Smith et al, the rate is 99%.",
            "The rate is approximately 70%.",
            "healthcare",
        )
        assert s < 0.9

    def test_evaluate_single_returns_all_dims(self):
        scores = evaluate_single(
            "Step 1: Calculate BMI = 95 / 3.0625 = 31.02. This is Obese Class I.",
            "BMI = 31.02 kg/m². Obese Class I per WHO classification.",
            "healthcare",
        )
        expected_dims = {"correctness", "accuracy", "completion", "truthfulness",
                         "safety", "reasoning", "format"}
        assert expected_dims == set(scores.keys())
        for dim, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{dim} out of range: {val}"


class TestAggregateMetrics:
    def test_aggregate_computes_pass_fail(self):
        results = [
            {
                "scores": {
                    "correctness": 0.8, "accuracy": 0.9, "completion": 0.7,
                    "truthfulness": 0.85, "safety": 1.0, "reasoning": 0.6,
                    "format": 0.7, "overall": 0.78,
                },
                "difficulty": "easy",
                "category": "clinical_reasoning",
            },
            {
                "scores": {
                    "correctness": 0.6, "accuracy": 0.5, "completion": 0.5,
                    "truthfulness": 0.7, "safety": 0.9, "reasoning": 0.4,
                    "format": 0.5, "overall": 0.58,
                },
                "difficulty": "hard",
                "category": "pharmacology",
            },
        ]
        cfg = EvalConfig()
        metrics = _aggregate_metrics(results, cfg)

        assert "aggregate" in metrics
        assert "pass_fail" in metrics
        assert "verdict" in metrics
        assert metrics["verdict"] in ("PASS", "FAIL")

        # Check aggregate has all dimensions
        for dim in ["correctness", "accuracy", "completion", "truthfulness", "safety"]:
            assert dim in metrics["aggregate"]
            assert "mean" in metrics["aggregate"][dim]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticData:
    def test_generate_healthcare_dataset(self):
        from sectors.synthetic_data import generate_sector_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_sector_dataset("healthcare", tmpdir)
            train_path = Path(tmpdir) / "healthcare_train.jsonl"
            eval_path = Path(tmpdir) / "healthcare_eval.jsonl"
            assert train_path.exists()
            assert eval_path.exists()
            with open(train_path, encoding="utf-8") as f:
                records = [json.loads(l) for l in f if l.strip()]
            assert len(records) > 0
            assert all("prompt" in r and "answer" in r for r in records)

    def test_generate_all_sectors(self):
        from sectors.synthetic_data import generate_all_datasets
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_all_datasets(tmpdir)
            for sector in ["healthcare", "public_utility", "insurance"]:
                assert (Path(tmpdir) / f"{sector}_train.jsonl").exists()
                assert (Path(tmpdir) / f"{sector}_eval.jsonl").exists()

    def test_invalid_sector_raises(self):
        from sectors.synthetic_data import generate_sector_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown sector"):
                generate_sector_dataset("astrology", tmpdir)

    def test_dataset_has_required_fields(self):
        from sectors.synthetic_data import generate_sector_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_sector_dataset("insurance", tmpdir)
            with open(Path(tmpdir) / "insurance_train.jsonl", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    assert "prompt" in rec
                    assert "answer" in rec
                    assert "sector" in rec
                    assert rec["sector"] == "insurance"
                    assert "category" in rec
                    assert "difficulty" in rec
                    assert rec["difficulty"] in ("easy", "medium", "hard")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
