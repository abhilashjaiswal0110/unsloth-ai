"""
Data Prep Agent: Validate Dataset Skill

Validates dataset quality, format compliance, and token length
distribution for fine-tuning with Unsloth.
"""

import json
from pathlib import Path


SUPPORTED_FORMATS = {
    "alpaca": {"required": ["instruction", "output"], "optional": ["input"]},
    "sharegpt": {"required": ["conversations"]},
    "raw_text": {"required": ["text"]},
    "preference": {"required": ["prompt", "chosen", "rejected"]},
}


def detect_format(sample: dict) -> str:
    """Auto-detect dataset format from a sample."""
    keys = set(sample.keys())

    if "conversations" in keys:
        return "sharegpt"
    if "instruction" in keys and "output" in keys:
        return "alpaca"
    if "prompt" in keys and "chosen" in keys and "rejected" in keys:
        return "preference"
    if "text" in keys:
        return "raw_text"

    return "unknown"


def validate_dataset(
    dataset_path: str,
    expected_format: str = None,
    max_seq_length: int = 2048,
    sample_size: int = 1000,
) -> dict:
    """
    Validate a dataset for fine-tuning readiness.

    Args:
        dataset_path: Path to JSONL file or HuggingFace dataset name
        expected_format: Expected format (auto-detected if None)
        max_seq_length: Maximum token sequence length
        sample_size: Number of samples to validate

    Returns:
        Validation report dictionary
    """
    report = {
        "total_samples": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "detected_format": None,
        "issues": [],
        "warnings": [],
        "stats": {},
    }

    # Load data
    data = []
    path = Path(dataset_path)
    if path.exists() and path.suffix in (".jsonl", ".json"):
        with open(path) as f:
            if path.suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            else:
                loaded = json.load(f)
                data = loaded if isinstance(loaded, list) else [loaded]
    else:
        report["issues"].append(f"File not found or unsupported format: {dataset_path}")
        return report

    report["total_samples"] = len(data)

    if not data:
        report["issues"].append("Dataset is empty")
        return report

    # Detect format
    detected = detect_format(data[0])
    report["detected_format"] = detected

    if expected_format and detected != expected_format:
        report["warnings"].append(
            f"Expected format '{expected_format}' but detected '{detected}'"
        )

    # Validate samples
    format_spec = SUPPORTED_FORMATS.get(detected, {})
    required_fields = format_spec.get("required", [])
    valid = 0
    invalid = 0
    text_lengths = []
    empty_fields = []

    samples_to_check = data[:sample_size]
    for i, sample in enumerate(samples_to_check):
        sample_valid = True

        # Check required fields
        for field in required_fields:
            if field not in sample:
                report["issues"].append(f"Sample {i}: missing required field '{field}'")
                sample_valid = False
            elif isinstance(sample[field], str) and not sample[field].strip():
                empty_fields.append(f"Sample {i}: empty field '{field}'")

        # Check text length
        if detected == "alpaca":
            text = f"{sample.get('instruction', '')} {sample.get('input', '')} {sample.get('output', '')}"
        elif detected == "raw_text":
            text = sample.get("text", "")
        elif detected == "sharegpt":
            convos = sample.get("conversations", [])
            text = " ".join(turn.get("value", "") for turn in convos)
        else:
            text = str(sample)

        word_count = len(text.split())
        text_lengths.append(word_count)

        if sample_valid:
            valid += 1
        else:
            invalid += 1

    report["valid_count"] = valid
    report["invalid_count"] = invalid

    if empty_fields:
        report["warnings"].append(
            f"Found {len(empty_fields)} samples with empty required fields"
        )

    # Compute statistics
    if text_lengths:
        report["stats"] = {
            "min_words": min(text_lengths),
            "max_words": max(text_lengths),
            "avg_words": sum(text_lengths) / len(text_lengths),
            "samples_checked": len(samples_to_check),
        }

    return report
