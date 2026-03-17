# Data Preparation Guide

This guide covers how to prepare, format, and validate datasets for fine-tuning with Unsloth.

## Table of Contents

- [Dataset Formats](#dataset-formats)
- [Alpaca Format](#alpaca-format)
- [ShareGPT / Conversational Format](#sharegpt--conversational-format)
- [Raw Text Format](#raw-text-format)
- [Preference Pairs (DPO)](#preference-pairs-dpo)
- [Using Unsloth Data Utilities](#using-unsloth-data-utilities)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Data Quality Best Practices](#data-quality-best-practices)

## Dataset Formats

Unsloth supports multiple dataset formats through HuggingFace's `datasets` library:

| Format | Use Case | Key Fields |
|--------|----------|------------|
| **Alpaca** | Instruction following | `instruction`, `input`, `output` |
| **ShareGPT** | Conversations | `conversations` (list of turns) |
| **Raw Text** | Continued pre-training | `text` |
| **Preference** | DPO/RLHF | `prompt`, `chosen`, `rejected` |

## Alpaca Format

The most common format for instruction fine-tuning.

### Schema

```json
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}
```

### Loading and Formatting

```python
from datasets import load_dataset

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Custom prompt template
alpaca_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

def format_alpaca(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        text = alpaca_template.format(
            instruction=examples["instruction"][i],
            input=examples["input"][i] or "",
            output=examples["output"][i],
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_alpaca, batched=True)
```

## ShareGPT / Conversational Format

For multi-turn conversations and chatbot training.

### Schema

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "Hello!"},
    {"from": "gpt", "value": "Hi! How can I help you today?"},
    {"from": "human", "value": "What is Python?"},
    {"from": "gpt", "value": "Python is a programming language..."}
  ]
}
```

### Loading and Formatting

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

dataset = load_dataset("your-sharegpt-dataset", split="train")

def format_conversations(examples):
    texts = []
    for convo in examples["conversations"]:
        # Convert ShareGPT format to model's chat template
        messages = []
        for turn in convo:
            role = "user" if turn["from"] == "human" else "assistant"
            if turn["from"] == "system":
                role = "system"
            messages.append({"role": role, "content": turn["value"]})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_conversations, batched=True)
```

## Raw Text Format

For continued pre-training on domain-specific text.

### Schema

```json
{
  "text": "Your raw text content here. This can be any length..."
}
```

### Loading

```python
from unsloth.dataprep import create_raw_text_dataset

# From text files
dataset = create_raw_text_dataset(
    tokenizer=tokenizer,
    text_files=["data/document1.txt", "data/document2.txt"],
    max_seq_length=2048,
)

# From HuggingFace
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
```

## Preference Pairs (DPO)

For Direct Preference Optimization training.

### Schema

```json
{
  "prompt": "Explain quantum computing.",
  "chosen": "Quantum computing uses quantum bits (qubits)...",
  "rejected": "Quantum computing is basically just faster computers..."
}
```

### Loading

```python
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences", split="train")

# Or create your own preference dataset
import json

preferences = [
    {
        "prompt": "How do I sort a list in Python?",
        "chosen": "You can sort a list in Python using:\n1. `sorted()` function...",
        "rejected": "just use sort lol"
    },
]

# Save as JSONL
with open("preferences.jsonl", "w") as f:
    for item in preferences:
        f.write(json.dumps(item) + "\n")

dataset = load_dataset("json", data_files="preferences.jsonl", split="train")
```

## Using Unsloth Data Utilities

### Synthetic Data Generation

Unsloth includes utilities for generating synthetic training data:

```python
from unsloth.dataprep.synthetic import generate_synthetic_data

# Generate instruction-following data
synthetic_dataset = generate_synthetic_data(
    config="synthetic_configs",
    num_samples=1000,
    seed=42,
)
```

### Raw Text Processing

```python
from unsloth.dataprep.raw_text import process_raw_text

# Process and chunk raw text files
processed_dataset = process_raw_text(
    files=["corpus.txt"],
    tokenizer=tokenizer,
    max_seq_length=2048,
    overlap=128,   # Sliding window overlap
)
```

## Creating Custom Datasets

### From CSV/Excel

```python
import pandas as pd
from datasets import Dataset

df = pd.read_csv("my_data.csv")  # Columns: question, answer

# Convert to instruction format
df["instruction"] = df["question"]
df["input"] = ""
df["output"] = df["answer"]

dataset = Dataset.from_pandas(df[["instruction", "input", "output"]])
```

### From JSON/JSONL

```python
dataset = load_dataset("json", data_files="my_data.jsonl", split="train")
```

### From a Database

```python
import sqlite3
import pandas as pd
from datasets import Dataset

conn = sqlite3.connect("knowledge_base.db")
df = pd.read_sql("SELECT question, answer FROM qa_pairs", conn)
conn.close()

dataset = Dataset.from_pandas(df)
```

### From Web Scraping Results

```python
data = [
    {"text": article_text}
    for article_text in scraped_articles
]
dataset = Dataset.from_list(data)
```

## Data Quality Best Practices

### 1. Clean Your Data

```python
def clean_text(example):
    text = example["text"]
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove HTML tags
    import re
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize Unicode
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    return {"text": text}

dataset = dataset.map(clean_text)
```

### 2. Deduplicate

```python
# Simple exact deduplication
seen = set()
def deduplicate(example):
    text_hash = hash(example["text"][:500])
    if text_hash in seen:
        return False
    seen.add(text_hash)
    return True

dataset = dataset.filter(deduplicate)
```

### 3. Filter by Quality

```python
def quality_filter(example):
    text = example["text"]
    # Minimum length
    if len(text.split()) < 10:
        return False
    # Maximum length
    if len(text.split()) > 10000:
        return False
    # Check for non-empty output
    if "output" in example and not example["output"].strip():
        return False
    return True

dataset = dataset.filter(quality_filter)
```

### 4. Validate Token Lengths

```python
def check_token_length(example):
    tokens = tokenizer(example["text"], truncation=False)
    return len(tokens["input_ids"]) <= max_seq_length

dataset = dataset.filter(check_token_length)
```

### 5. Balance Your Dataset

```python
# If you have categories, ensure balanced representation
from collections import Counter

categories = Counter(dataset["category"])
min_count = min(categories.values())

balanced_data = []
for cat in categories:
    cat_data = dataset.filter(lambda x: x["category"] == cat)
    balanced_data.append(cat_data.select(range(min_count)))

from datasets import concatenate_datasets
balanced_dataset = concatenate_datasets(balanced_data).shuffle(seed=42)
```

## Dataset Size Guidelines

| Dataset Size | Training Time | Recommended Epochs | Use Case |
|-------------|--------------|-------------------|----------|
| 100-1K | Minutes | 3-10 | Quick experiments, PoC |
| 1K-10K | 30min-2hrs | 2-5 | Domain-specific tasks |
| 10K-100K | 2-12hrs | 1-3 | General instruction following |
| 100K-1M | 12-48hrs | 1-2 | Large-scale fine-tuning |

## Troubleshooting

### Common Issues

**"Tokenizer max length exceeded"**
- Solution: Set `max_seq_length` appropriately and enable truncation

**"Dataset format not recognized"**
- Solution: Ensure your dataset has the required fields for your chosen format

**"Out of memory during data loading"**
- Solution: Use `streaming=True` with `load_dataset()` for large datasets

## Next Steps

- [Fine-Tuning Guide](FINE_TUNING_GUIDE.md) — Use your prepared data for training
- [Examples](EXAMPLES.md) — More dataset preparation examples
