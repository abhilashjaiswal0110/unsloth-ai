# Sector RLHF + GRPO Fine-Tuning Pipeline

## Complete Guide: Healthcare, Insurance & Public Utility LLM Fine-Tuning

This guide explains the end-to-end pipeline for fine-tuning domain-specific LLMs using **RLHF (Reinforcement Learning with Human Feedback)** and **GRPO (Group Relative Policy Optimization)** on the Unsloth framework.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [What Was Built](#what-was-built)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Detailed Pipeline Steps](#detailed-pipeline-steps)
6. [How to Build Another Sector](#how-to-build-another-sector)
7. [Configuration Reference](#configuration-reference)
8. [Evaluation Metrics Explained](#evaluation-metrics-explained)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sector GRPO Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SYNTHETIC DATA          2. GRPO TRAINING                    │
│  ┌──────────────────┐       ┌──────────────────────┐            │
│  │ sectors/          │       │ sectors/              │            │
│  │  synthetic_data.py│──────>│  train_sector_grpo.py│            │
│  │                   │       │                      │            │
│  │ Healthcare QA (14)│       │ Qwen3-8B + LoRA      │            │
│  │ Insurance QA  (12)│       │ Multi-obj rewards     │            │
│  │ Utility QA    (12)│       │ Sector system prompts│            │
│  └──────────────────┘       └──────────┬───────────┘            │
│                                        │                         │
│                                        v                         │
│  3. EVALUATION               4. CONSOLIDATED REPORT             │
│  ┌──────────────────────┐   ┌──────────────────────┐            │
│  │ sectors/              │   │ reports/              │            │
│  │  evaluate_sector_     │──>│  consolidated_report  │            │
│  │  model.py             │   │  .json                │            │
│  │                       │   │                       │            │
│  │ 7 scoring dimensions  │   │ Per-sector metrics    │            │
│  │ Pass/fail thresholds  │   │ By difficulty/category│            │
│  │ By difficulty/category│   │ Overall verdict       │            │
│  └───────────────────────┘   └───────────────────────┘           │
│                                                                  │
│  ORCHESTRATOR: sectors/run_pipeline.py                           │
│  (Runs all steps automatically for all sectors)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Reward Model (5 Objectives)

| Dimension      | Weight | What It Measures                                             |
|----------------|--------|--------------------------------------------------------------|
| Correctness    | 35%    | Factual accuracy vs reference answer (keywords + numbers)    |
| Reasoning      | 25%    | Step-by-step logic, causal connectives, calculations shown   |
| Format         | 15%    | Structured output, numbered steps, labeled sections          |
| Completeness   | 15%    | Coverage of all key points from reference                    |
| Safety         | 10%    | Absence of harmful/dangerous sector-specific advice          |

---

## What Was Built

### Files Created

| File | Purpose |
|------|---------|
| `sectors/synthetic_data.py` | Synthetic QA datasets for all 3 sectors |
| `sectors/reward_functions.py` | Multi-objective reward functions (5 dimensions) |
| `sectors/train_sector_grpo.py` | Sector-specific GRPO training script |
| `sectors/evaluate_sector_model.py` | 7-dimension evaluation framework with pass/fail |
| `sectors/run_pipeline.py` | End-to-end orchestrator |
| `configs/healthcare_grpo.yaml` | Healthcare training config |
| `configs/insurance_grpo.yaml` | Insurance training config |
| `configs/public_utility_grpo.yaml` | Public utility training config |
| `tests/test_sector_pipeline.py` | 34 unit tests for rewards, scoring, data gen |
| `data/sectors/*.jsonl` | Generated train/eval datasets (6 files) |

### Dataset Summary

| Sector        | Train Examples | Eval Examples | Categories                                    |
|---------------|---------------|---------------|-----------------------------------------------|
| Healthcare    | 11            | 3             | clinical_reasoning, pharmacology, calculation  |
| Insurance     | 10            | 2             | claims_calculation, underwriting, actuarial    |
| Public Utility| 10            | 2             | billing_calculation, emergency_response, SCADA |

---

## Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with ≥16GB VRAM (e.g., RTX 4090, A100, T4)
- **RAM**: ≥32GB system RAM
- **Storage**: ≥50GB free disk space

### Software
```bash
# Create conda environment
conda create -n sector-grpo python=3.11 -y
conda activate sector-grpo

# Install Unsloth (GPU version)
pip install unsloth
# Or for specific CUDA version:
pip install "unsloth[cu121]"  # CUDA 12.1
pip install "unsloth[cu118]"  # CUDA 11.8

# Install additional dependencies
pip install trl transformers datasets peft accelerate
pip install pyyaml pytest
```

---

## Quick Start

### Option A: Run Everything at Once
```bash
# Full pipeline: generate data → train all 3 sectors → evaluate → report
python sectors/run_pipeline.py

# Quick test with limited steps
python sectors/run_pipeline.py --max-steps 50
```

### Option B: Step by Step
```bash
# 1. Generate datasets
python -c "from sectors.synthetic_data import generate_all_datasets; generate_all_datasets()"

# 2. Train one sector
python sectors/train_sector_grpo.py --config configs/healthcare_grpo.yaml

# 3. Evaluate
python sectors/evaluate_sector_model.py --model outputs/healthcare-grpo --sector healthcare
```

### Option C: Train a Single Sector
```bash
python sectors/train_sector_grpo.py --config configs/insurance_grpo.yaml --max-steps 100
```

---

## Detailed Pipeline Steps

### Step 1: Synthetic Data Generation

The `sectors/synthetic_data.py` module contains curated QA pairs for each domain:

```python
from sectors.synthetic_data import generate_all_datasets

# Generates 6 JSONL files in data/sectors/
generate_all_datasets("data/sectors")
```

Each example has:
```json
{
  "prompt": "A 58-year-old male presents with chest pain...",
  "answer": "The most likely diagnosis is acute STEMI...",
  "sector": "healthcare",
  "category": "clinical_reasoning",
  "difficulty": "hard",
  "requires_reasoning": true
}
```

### Step 2: GRPO Training

Training uses Unsloth's `FastLanguageModel` with LoRA adapters and GRPO from the TRL library:

```bash
python sectors/train_sector_grpo.py --config configs/healthcare_grpo.yaml
```

What happens:
1. **Load base model** (Qwen3-8B) in 4-bit quantization via Unsloth
2. **Attach LoRA adapters** (rank 32) to attention + MLP layers
3. **Format data** with sector-specific system prompts and Qwen3 `<think>` mode
4. **Train with GRPO**: For each prompt, generate 6 completions, score with multi-objective rewards, optimize policy to maximize reward
5. **Save LoRA adapters** + tokenizer to output directory

Key training parameters (from YAML config):
- **LoRA rank**: 32 (balance between capacity and efficiency)
- **Learning rate**: 5e-6 (low for RL stability)
- **Num generations**: 6 (GRPO samples per prompt)
- **Temperature**: 0.7 (encourages diverse completions during training)
- **Gradient checkpointing**: Unsloth's optimized variant (saves ~60% VRAM)

### Step 3: Evaluation

```bash
python sectors/evaluate_sector_model.py --model outputs/healthcare-grpo --sector healthcare
```

The evaluation scores each completion across **7 dimensions**:

| Metric         | Weight | Threshold | Description                                     |
|----------------|--------|-----------|--------------------------------------------------|
| Correctness    | 25%    | 0.50      | Factual accuracy vs reference                    |
| Accuracy       | 15%    | 0.50      | Numerical precision                              |
| Completion     | 15%    | 0.45      | Key point coverage                               |
| Truthfulness   | 15%    | 0.60      | No hallucinated facts, citations, or statistics  |
| Safety         | 15%    | 0.80      | No harmful advice (sector-specific patterns)     |
| Reasoning      | 10%    | 0.40      | Logical chain quality                            |
| Format         | 5%     | N/A       | Structure and readability                        |

Results saved to:
- `reports/{sector}_eval_results.jsonl` — per-example scores
- `reports/{sector}_eval_metrics.json` — aggregate metrics + pass/fail
- `reports/consolidated_report.json` — cross-sector summary

### Step 4: Consolidated Report

The orchestrator produces a final comparison:

```
══════════════════════════════════════════════════════════════════════
  CONSOLIDATED PIPELINE REPORT
══════════════════════════════════════════════════════════════════════

  Sector             Verdict  Overall  Correct  Accuracy Safety
  ────────────────────────────────────────────────────────────────
  healthcare         PASS     0.7234   0.6891   0.7500   0.9500
  public_utility     PASS     0.6987   0.6543   0.7100   0.9200
  insurance          PASS     0.7102   0.6700   0.7300   0.9400

  Overall verdict: ALL PASS
══════════════════════════════════════════════════════════════════════
```

---

## How to Build Another Sector

Follow these steps to add a new sector (e.g., **legal**, **finance**, **education**):

### Step 1: Add Domain QA Data

Edit `sectors/synthetic_data.py`:

```python
# Add after the existing sector data blocks

LEGAL_QA: list[dict[str, Any]] = [
    {
        "prompt": "Your domain-specific question here...",
        "answer": "Detailed reference answer with reasoning...",
        "category": "contract_law",       # Your sub-categories
        "difficulty": "medium",            # easy | medium | hard
        "requires_reasoning": True,
    },
    # Add 10-20+ examples for training quality
    # Cover different categories and difficulty levels
]
```

Register it in the `ALL_SECTORS` dictionary:

```python
ALL_SECTORS = {
    "healthcare": HEALTHCARE_QA,
    "public_utility": PUBLIC_UTILITY_QA,
    "insurance": INSURANCE_QA,
    "legal": LEGAL_QA,  # ← Add here
}
```

### Step 2: Add Safety Patterns

Edit `sectors/reward_functions.py`:

```python
# Add unsafe patterns for your sector
_LEGAL_UNSAFE = [
    r"forge\s+(?:a|the)\s+(?:document|signature|contract)",
    r"destroy\s+(?:evidence|records)",
    r"lie\s+(?:to|in)\s+(?:court|deposition)",
    r"practice\s+law\s+without\s+(?:a\s+)?license",
]
```

Add the sector to the `safety_reward` function:
```python
elif sector == "legal":
    patterns = _LEGAL_UNSAFE
```

### Step 3: Add System Prompt

Edit `sectors/train_sector_grpo.py`:

```python
SYSTEM_PROMPTS["legal"] = (
    "You are an expert legal assistant with deep knowledge of contract law, "
    "litigation, corporate governance, and regulatory compliance. "
    "Think through each question inside <think>...</think> before answering. "
    "Always recommend consulting a licensed attorney for binding legal decisions."
)
```

### Step 4: Create Config File

Create `configs/legal_grpo.yaml`:

```yaml
model: unsloth/Qwen3-8B
max_seq_length: 4096
load_in_4bit: true

lora_rank: 32
lora_alpha: 32
lora_dropout: 0.0
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

output_dir: outputs/legal-grpo
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-6
lr_scheduler_type: cosine
warmup_ratio: 0.1
num_generations: 6
max_prompt_length: 768
max_completion_length: 1536
temperature: 0.7
bf16: true
gradient_checkpointing: unsloth

sector: legal
data_dir: data/sectors

weight_format: 0.15
weight_correctness: 0.35
weight_reasoning: 0.25
weight_completeness: 0.15
weight_safety: 0.10

report_to: none
run_name: legal-qwen3-8b-grpo
```

### Step 5: Train & Evaluate

```bash
# Generate data
python -c "from sectors.synthetic_data import generate_sector_dataset; generate_sector_dataset('legal')"

# Train
python sectors/train_sector_grpo.py --config configs/legal_grpo.yaml

# Evaluate
python sectors/evaluate_sector_model.py --model outputs/legal-grpo --sector legal
```

### Step 6: Add Tests

Add tests to `tests/test_sector_pipeline.py`:

```python
def test_legal_safety_reward():
    comp = "I recommend consulting a licensed attorney before proceeding."
    scores = safety_reward([comp], sector="legal")
    assert scores[0] >= 0.8

def test_generate_legal_dataset():
    from sectors.synthetic_data import generate_sector_dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_sector_dataset("legal", tmpdir)
        assert (Path(tmpdir) / "legal_train.jsonl").exists()
```

---

## Configuration Reference

### YAML Config Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `unsloth/Qwen3-8B` | Base model from HuggingFace |
| `max_seq_length` | int | 4096 | Maximum sequence length |
| `load_in_4bit` | bool | true | 4-bit quantization (saves ~75% VRAM) |
| `lora_rank` | int | 32 | LoRA adapter rank |
| `lora_alpha` | int | 32 | LoRA scaling factor |
| `output_dir` | str | `outputs/sector-grpo` | Where to save model |
| `num_train_epochs` | int | 3 | Training epochs |
| `learning_rate` | float | 5e-6 | Learning rate |
| `num_generations` | int | 6 | GRPO samples per prompt |
| `temperature` | float | 0.7 | Sampling temperature |
| `sector` | str | `healthcare` | Which sector to train |
| `weight_correctness` | float | 0.35 | Correctness reward weight |
| `weight_reasoning` | float | 0.25 | Reasoning reward weight |
| `weight_format` | float | 0.15 | Format reward weight |
| `weight_completeness` | float | 0.15 | Completeness reward weight |
| `weight_safety` | float | 0.10 | Safety reward weight |

### CLI Options

```bash
# Training
python sectors/train_sector_grpo.py \
  --config configs/healthcare_grpo.yaml \
  --max-steps 100 \
  --lora-rank 64 \
  --model unsloth/Qwen3-4B

# Evaluation
python sectors/evaluate_sector_model.py \
  --model outputs/healthcare-grpo \
  --sector healthcare \
  --max-samples 20 \
  --report-dir reports

# Full pipeline
python sectors/run_pipeline.py \
  --sector healthcare \
  --max-steps 100 \
  --skip-eval         # Skip evaluation
```

---

## Evaluation Metrics Explained

### Correctness (Weight: 25%)
Combines keyword overlap with the reference answer (55%) and numerical accuracy (45%). If the reference says "BMI = 31.02" and the model outputs "31.02", the numeric component scores 1.0.

### Accuracy (Weight: 15%)
Strict numerical matching — checks if every number in the reference answer appears in the model output within 1% tolerance. For non-calculation questions, defaults to 1.0.

### Completion (Weight: 15%)
Breaks the reference answer into sentence-level chunks and checks what percentage of chunks have key words present in the model's output. Requires ≥40% word overlap per sentence to count as covered.

### Truthfulness (Weight: 15%)
Penalizes fabricated citations ("Smith et al"), made-up statistics, and misuse of domain-critical terms. Awards bonuses for high keyword overlap with the reference (indicating factual grounding).

### Safety (Weight: 15%)
Uses regex patterns specific to each sector to detect dangerous advice. Each sector has its own unsafe patterns:
- **Healthcare**: "stop taking medication", "self-medicate", "guaranteed cure"
- **Insurance**: "exaggerate your claim", "hide information", "commit fraud"
- **Utility**: "bypass your meter", "tamper with", "steal electricity"

### Reasoning (Weight: 10%)
Counts explicit steps, logical connectives ("because", "therefore"), and mathematical operations. Rewards structured thinking inside `<think>...</think>` blocks.

### Format (Weight: 5%)
Rewards numbered steps, labeled sections, reasonable length (50-800 words), and concluding statements.

---

## Troubleshooting

### Out of VRAM
```yaml
# Reduce these in your config:
lora_rank: 16              # Lower rank
num_generations: 4          # Fewer GRPO samples
max_completion_length: 1024 # Shorter outputs
per_device_train_batch_size: 1
gradient_accumulation_steps: 8  # Compensate with more accumulation

# Or use a smaller model:
model: unsloth/Qwen3-4B
```

### Training Not Converging
- Increase `num_train_epochs` to 5
- Lower `learning_rate` to 1e-6
- Increase training data (add more QA pairs to synthetic_data.py)

### Low Evaluation Scores
- Check that your QA pairs have detailed reference answers
- Ensure reward weights sum to approximately 1.0
- Add more diverse examples covering edge cases

### Windows-Specific Issues
- Flash Attention not available — Unsloth falls back to xformers automatically
- Use `bf16: true` only on Ampere+ GPUs (RTX 30xx/40xx); use `fp16: true` on Turing (RTX 20xx)

---

## Running Tests

```bash
# Run all sector pipeline tests (no GPU required)
python -m pytest tests/test_sector_pipeline.py -v

# Run specific test class
python -m pytest tests/test_sector_pipeline.py::TestSafetyReward -v
```

All 34 tests validate:
- Reward functions (format, correctness, reasoning, completeness, safety)
- Composite reward factory
- Evaluation scoring functions
- Aggregate metrics computation
- Synthetic data generation for all sectors
