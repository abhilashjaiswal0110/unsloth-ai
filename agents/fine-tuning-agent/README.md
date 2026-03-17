# Fine-Tuning Agent

An intelligent workflow agent that guides users through the complete model fine-tuning process with Unsloth.

## Overview

The Fine-Tuning Agent handles the end-to-end fine-tuning workflow:

1. **Model Selection** — Recommends the best model based on your GPU and use case
2. **LoRA Configuration** — Optimizes LoRA hyperparameters for your setup
3. **Training Setup** — Configures training arguments with best practices
4. **Monitoring** — Tracks training progress and detects issues
5. **Saving** — Saves the model in your preferred format

## Modes

| Mode | Description |
|------|-------------|
| `quick` | Minimal configuration, sensible defaults |
| `guided` | Interactive step-by-step walkthrough |
| `advanced` | Full control over all parameters |
| `benchmark` | Fine-tune with evaluation metrics |

## Usage

### Quick Mode

```python
from agents.fine_tuning_agent.skills.recommend_config import recommend_config
from agents.fine_tuning_agent.skills.generate_script import generate_training_script

# Get recommended configuration
config = recommend_config(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    dataset_name="yahma/alpaca-cleaned",
    gpu_vram_gb=24,
    task_type="instruction",
)

# Generate and run training script
script = generate_training_script(config)
```

### AI Assistant Prompt

```
I want to fine-tune a model for customer support. Help me:
1. Choose the best model for my RTX 3090 (24GB)
2. Configure LoRA parameters
3. Set up training with my dataset at data/support_tickets.jsonl
4. Save as GGUF for Ollama
```

## Skills

| Skill | Purpose |
|-------|---------|
| `recommend_config` | Recommend model and hyperparameters based on constraints |
| `generate_script` | Generate a complete training script |
| `validate_config` | Validate configuration before training |
| `monitor_training` | Track loss curves and detect anomalies |
