# Reinforcement Learning Training Guide

This guide covers reinforcement learning (RL) training with Unsloth, focusing on GRPO (Group Relative Policy Optimization).

## Table of Contents

- [Overview](#overview)
- [GRPO Training](#grpo-training)
- [Use Case: Math Reasoning](#use-case-math-reasoning)
- [Use Case: Code Correctness](#use-case-code-correctness)
- [Use Case: Safety Alignment](#use-case-safety-alignment)
- [DPO Training](#dpo-training)
- [Custom Reward Functions](#custom-reward-functions)
- [Best Practices](#best-practices)

## Overview

Reinforcement learning fine-tuning trains models to optimize specific reward signals rather than just imitating training data. Unsloth supports:

- **GRPO** — Group Relative Policy Optimization (recommended)
- **DPO** — Direct Preference Optimization
- **Online DPO** — Interactive preference learning

## GRPO Training

GRPO generates multiple responses per prompt, scores them with reward functions, and updates the model to favor higher-scoring outputs.

### Basic GRPO Setup

```python
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
)

# Define reward function
def reward_function(completions, **kwargs):
    """Score responses based on quality criteria."""
    scores = []
    for completion in completions:
        score = 0.0
        # Length reward: prefer concise responses
        if len(completion) < 500:
            score += 1.0
        # Quality signals
        if "therefore" in completion.lower() or "because" in completion.lower():
            score += 0.5  # Reward reasoning
        scores.append(score)
    return scores

# Load dataset with prompts
dataset = load_dataset("your-dataset", split="train")

# Configure GRPO
training_args = GRPOConfig(
    output_dir="outputs/grpo-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    max_completion_length=512,
    num_generations=4,          # Generate 4 responses per prompt
    logging_steps=1,
    fp16=True,
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_funcs=[reward_function],
    args=training_args,
)

trainer.train()
model.save_pretrained("grpo-finetuned-model")
```

## Use Case: Math Reasoning

**Goal**: Train a model that produces correct mathematical reasoning with verified answers.

```python
import re

def math_reward(completions, answers, **kwargs):
    """Reward correct mathematical answers."""
    scores = []
    for completion, expected in zip(completions, answers):
        score = 0.0

        # Extract numerical answer from response
        numbers = re.findall(r"\\boxed\{(.+?)\}", completion)
        if not numbers:
            numbers = re.findall(r"(?:answer|result)\s*(?:is|=)\s*(\d+\.?\d*)", completion)

        if numbers and str(expected) in numbers[-1]:
            score += 2.0   # Correct answer

        # Reward step-by-step reasoning
        steps = completion.count("Step")
        score += min(steps * 0.1, 0.5)

        # Reward showing work
        if "=" in completion:
            score += 0.3

        scores.append(score)
    return scores

# Dataset format: {"prompt": "What is 15% of 200?", "answer": "30"}
dataset = load_dataset("gsm8k", "main", split="train")

training_args = GRPOConfig(
    output_dir="outputs/math-grpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=2,
    max_completion_length=1024,
    num_generations=8,
    logging_steps=1,
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_funcs=[math_reward],
    args=training_args,
)

trainer.train()
```

### Example Prompts for Math Reasoning

```
Solve the following math problem step by step:
A store offers a 20% discount on a $150 jacket. If sales tax is 8%, what is the final price?
```

```
Calculate: If a train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours, what is the total distance covered?
```

## Use Case: Code Correctness

**Goal**: Train a model that generates code which passes test cases.

```python
import subprocess
import tempfile

def code_execution_reward(completions, test_cases, **kwargs):
    """Reward code that passes test cases."""
    scores = []
    for completion, tests in zip(completions, test_cases):
        score = 0.0

        # Extract code block
        code_match = re.search(r"```python\n(.*?)```", completion, re.DOTALL)
        if not code_match:
            scores.append(-1.0)
            continue

        code = code_match.group(1)
        full_code = code + "\n" + tests

        # Execute safely in sandbox
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
                f.write(full_code)
                f.flush()
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    score = 2.0   # All tests pass
                else:
                    score = -0.5  # Syntax/runtime error
        except subprocess.TimeoutExpired:
            score = -0.5
        except Exception:
            score = -1.0

        # Reward clean code structure
        if "def " in code:
            score += 0.2
        if '"""' in code or "'''" in code:
            score += 0.1  # Docstrings

        scores.append(score)
    return scores
```

## Use Case: Safety Alignment

**Goal**: Train a model that is helpful while refusing harmful requests.

```python
UNSAFE_PATTERNS = [
    "how to hack", "create malware", "make a bomb",
    "steal data", "bypass security",
]

def safety_reward(completions, is_safe_prompt, **kwargs):
    """Reward safe, helpful responses."""
    scores = []
    for completion, safe in zip(completions, is_safe_prompt):
        score = 0.0

        if safe:
            # For safe prompts, reward helpfulness
            if len(completion) > 50:
                score += 1.0
            if any(w in completion.lower() for w in ["here's", "sure", "certainly"]):
                score += 0.5
        else:
            # For unsafe prompts, reward refusal
            if any(w in completion.lower() for w in
                   ["i cannot", "i can't", "not appropriate", "sorry"]):
                score += 2.0
            else:
                score -= 2.0  # Penalize unsafe compliance

        scores.append(score)
    return scores
```

## DPO Training

DPO trains on preference pairs (chosen vs rejected responses).

```python
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

PatchDPOTrainer()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
)

# Dataset with chosen/rejected pairs
# Format: {"prompt": "...", "chosen": "Good response", "rejected": "Bad response"}
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences", split="train")

training_args = DPOConfig(
    output_dir="outputs/dpo-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    beta=0.1,                    # KL penalty coefficient
    logging_steps=1,
    fp16=True,
)

trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
```

## Custom Reward Functions

### Multi-Objective Reward

```python
def multi_objective_reward(completions, **kwargs):
    """Combine multiple reward signals."""
    scores = []
    for completion in completions:
        # Factuality: reward citations and references
        factuality = 0.5 if "[source]" in completion or "according to" in completion.lower() else 0.0

        # Conciseness: penalize overly long responses
        length = len(completion.split())
        conciseness = 1.0 if length < 200 else max(0, 1.0 - (length - 200) / 500)

        # Structure: reward organized responses
        structure = 0.0
        if any(marker in completion for marker in ["1.", "•", "- ", "First,"]):
            structure += 0.3
        if "\n\n" in completion:
            structure += 0.2

        total = factuality + conciseness + structure
        scores.append(total)
    return scores
```

### Format-Enforcing Reward

```python
def json_format_reward(completions, **kwargs):
    """Reward valid JSON output."""
    import json
    scores = []
    for completion in completions:
        try:
            parsed = json.loads(completion)
            score = 1.0
            # Bonus for expected keys
            if isinstance(parsed, dict):
                expected_keys = {"name", "description", "category"}
                found = expected_keys & set(parsed.keys())
                score += len(found) * 0.5
        except json.JSONDecodeError:
            score = -1.0
        scores.append(score)
    return scores
```

## Best Practices

### GRPO Tips

1. **Start with small `num_generations`** (4-8) and increase if reward signal is noisy
2. **Use multiple reward functions** to balance different objectives
3. **Monitor KL divergence** to prevent the model from diverging too far
4. **Use a lower learning rate** (1e-6 to 5e-6) than SFT
5. **Validate on held-out prompts** regularly

### DPO Tips

1. **High-quality preference pairs** are critical — noisy labels degrade performance
2. **Beta parameter** controls preference strength (0.1-0.5 typical)
3. **Reference model** should be the base model before DPO training

### Common Pitfalls

- **Reward hacking**: Model finds shortcuts to maximize reward without genuine improvement
- **Mode collapse**: Model produces identical responses — increase temperature or diversity penalty
- **Training instability**: Reduce learning rate or increase gradient accumulation steps

## Next Steps

- [Fine-Tuning Guide](FINE_TUNING_GUIDE.md) — SFT training workflows
- [Data Preparation](DATA_PREPARATION.md) — Build custom RL datasets
- [Examples](EXAMPLES.md) — More RL training examples
