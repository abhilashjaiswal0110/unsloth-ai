# Data Preparation Agent

An intelligent agent for preparing, validating, and optimizing datasets for Unsloth fine-tuning.

## Overview

The Data Preparation Agent automates the data pipeline:

1. **Format Detection** — Auto-detect dataset format (Alpaca, ShareGPT, raw text)
2. **Validation** — Check for common data quality issues
3. **Transformation** — Convert between formats
4. **Cleaning** — Remove duplicates, fix encoding, normalize text
5. **Analysis** — Token distribution, quality metrics

## Modes

| Mode | Description |
|------|-------------|
| `validate` | Check dataset quality without modifications |
| `transform` | Convert between dataset formats |
| `clean` | Clean and deduplicate data |
| `analyze` | Generate dataset statistics and visualizations |

## Usage

### Validate a Dataset

```python
from agents.data_prep_agent.skills.validate_dataset import validate_dataset

report = validate_dataset(
    dataset_path="data/training_data.jsonl",
    expected_format="alpaca",
    tokenizer_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
)

print(f"Valid samples: {report['valid_count']}")
print(f"Issues found: {report['issues']}")
```

### Clean a Dataset

```python
from agents.data_prep_agent.skills.clean_dataset import clean_dataset

cleaned = clean_dataset(
    dataset_path="data/raw_data.jsonl",
    remove_duplicates=True,
    fix_encoding=True,
    min_length=10,
    max_length=10000,
)

cleaned.save_to_disk("data/cleaned_data")
```

### AI Assistant Prompt

```
I have a CSV file with columns 'question' and 'answer'. Help me:
1. Convert it to Alpaca format for fine-tuning
2. Validate the data quality
3. Check token lengths for Llama 3.2 3B
4. Remove any duplicates or low-quality entries
```

## Skills

| Skill | Purpose |
|-------|---------|
| `validate_dataset` | Check data quality and format compliance |
| `clean_dataset` | Clean, deduplicate, and normalize data |
| `convert_format` | Convert between Alpaca, ShareGPT, and raw text formats |
| `analyze_dataset` | Generate statistics on token lengths, quality metrics |
