"""
Data Prep Agent: Clean Dataset Skill

Cleans and deduplicates datasets for fine-tuning.
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path


def clean_text(text: str) -> str:
    """Clean and normalize a text string."""
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace (preserve newlines)
    lines = text.split("\n")
    lines = [" ".join(line.split()) for line in lines]
    text = "\n".join(lines)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def clean_dataset(
    dataset_path: str,
    output_path: str = None,
    remove_duplicates: bool = True,
    fix_encoding: bool = True,
    min_length: int = 10,
    max_length: int = 50000,
) -> dict:
    """
    Clean a dataset file.

    Args:
        dataset_path: Path to input JSONL file
        output_path: Path for cleaned output (default: adds '_cleaned' suffix)
        remove_duplicates: Whether to remove exact duplicates
        fix_encoding: Whether to fix Unicode encoding issues
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters
    Returns:
        Cleaning report dictionary
    """
    path = Path(dataset_path)
    if output_path is None:
        output_path = str(path.with_stem(path.stem + "_cleaned"))

    report = {
        "input_count": 0,
        "output_count": 0,
        "duplicates_removed": 0,
        "too_short": 0,
        "too_long": 0,
        "encoding_fixed": 0,
    }

    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    report["input_count"] = len(data)

    # Deduplicate
    seen_hashes = set()
    cleaned = []

    for sample in data:
        # Create content hash for deduplication
        content_key = json.dumps(sample, sort_keys=True)
        content_hash = hashlib.md5(content_key.encode()).hexdigest()

        if remove_duplicates and content_hash in seen_hashes:
            report["duplicates_removed"] += 1
            continue
        seen_hashes.add(content_hash)

        # Clean text fields
        cleaned_sample = {}
        for key, value in sample.items():
            if isinstance(value, str):
                if fix_encoding:
                    original = value
                    value = clean_text(value)
                    if value != original:
                        report["encoding_fixed"] += 1

                # Check length
                if len(value) < min_length and key in ("output", "text", "answer"):
                    report["too_short"] += 1
                    break
                if len(value) > max_length:
                    report["too_long"] += 1
                    break

            cleaned_sample[key] = value
        else:
            cleaned.append(cleaned_sample)

    report["output_count"] = len(cleaned)

    # Write output
    with open(output_path, "w") as f:
        for sample in cleaned:
            f.write(json.dumps(sample) + "\n")

    return report
