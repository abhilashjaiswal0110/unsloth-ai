# Model Evaluation Agent

An agent for evaluating fine-tuned model quality, comparing against baselines, and running benchmarks.

## Overview

The Model Evaluation Agent helps assess model quality:

1. **Inference Testing** — Generate responses and inspect quality
2. **Benchmark Evaluation** — Run standardized benchmarks
3. **Comparison** — Compare fine-tuned vs base model
4. **Regression Testing** — Ensure the model hasn't degraded on core capabilities

## Modes

| Mode | Description |
|------|-------------|
| `quick` | Fast spot-check with a few prompts |
| `benchmark` | Run standard benchmarks (MMLU, GSM8K, etc.) |
| `compare` | Side-by-side comparison with base model |
| `regression` | Test against a suite of expected behaviors |

## Usage

### Quick Evaluation

```python
from agents.model_eval_agent.skills.evaluate_model import quick_eval

results = quick_eval(
    model_path="./my-finetuned-model",
    test_prompts=[
        "Explain machine learning in simple terms.",
        "Write a Python function to reverse a string.",
        "What is the capital of France?",
    ],
)

for prompt, response in results.items():
    print(f"Prompt: {prompt}")
    print(f"Response: {response[:200]}...")
    print("---")
```

### Compare Models

```python
from agents.model_eval_agent.skills.compare_models import compare_models

comparison = compare_models(
    base_model="unsloth/Llama-3.2-3B-Instruct",
    finetuned_model="./my-finetuned-model",
    test_prompts=["Explain quantum computing."],
)
```

### AI Assistant Prompt

```
Evaluate my fine-tuned model at ./customer-support-model:
1. Test with 10 customer support scenarios
2. Compare responses against the base Llama 3.2 3B
3. Check for response quality, accuracy, and tone
4. Identify any capability regressions
```

## Skills

| Skill | Purpose |
|-------|---------|
| `evaluate_model` | Run inference and assess response quality |
| `compare_models` | Side-by-side base vs fine-tuned comparison |
| `run_benchmark` | Execute standardized benchmarks |
