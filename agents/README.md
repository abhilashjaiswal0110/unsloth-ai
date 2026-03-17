# Unsloth AI Agents

This directory contains AI agent skills and plugins for automating common Unsloth workflows. Each agent is a self-contained module that guides users through specific tasks using structured prompts and validation.

## Available Agents

| Agent | Purpose | Use Case |
|-------|---------|----------|
| [fine-tuning-agent](fine-tuning-agent/) | Guided model fine-tuning | Select model, configure LoRA, train, and save |
| [data-prep-agent](data-prep-agent/) | Dataset preparation & validation | Format, clean, validate datasets for training |
| [model-eval-agent](model-eval-agent/) | Model evaluation & benchmarking | Evaluate model quality, compare baselines |
| [export-agent](export-agent/) | Model export & deployment | Export to GGUF, vLLM, HuggingFace Hub |
| [rl-training-agent](rl-training-agent/) | Reinforcement learning training | GRPO, DPO, custom reward functions |

## Architecture

Each agent follows a common structure:

```
agent-name/
├── README.md           # Agent documentation & usage
├── agent.yaml          # Agent configuration and metadata
├── skills/             # Composable skill modules
│   ├── skill_name.py   # Individual skill implementation
│   └── ...
└── prompts/            # Prompt templates (if applicable)
```

## How Agents Work

Agents are **workflow orchestrators** that:

1. **Assess** the user's environment (GPU, VRAM, installed packages)
2. **Recommend** optimal configurations based on constraints
3. **Generate** ready-to-run code with best practices
4. **Validate** outputs and catch common mistakes
5. **Guide** through multi-step processes with checkpoints

## Usage

### As Python Modules

```python
from agents.fine_tuning_agent import FineTuningAgent

agent = FineTuningAgent()

# Get recommended configuration
config = agent.recommend_config(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    dataset_name="yahma/alpaca-cleaned",
    gpu_vram_gb=24,
)

# Generate training script
script = agent.generate_training_script(config)
print(script)
```

### As CLI Workflows

```bash
# Interactive fine-tuning wizard
python -m agents.fine_tuning_agent

# Data preparation pipeline
python -m agents.data_prep_agent --input data.jsonl --format alpaca

# Model evaluation
python -m agents.model_eval_agent --model ./my-model --benchmark mmlu

# Export workflow
python -m agents.export_agent --model ./my-model --format gguf --quantization q4_k_m
```

### With Claude / AI Assistants

The agents include structured prompts that can be used with AI coding assistants:

```
Use the fine-tuning-agent to help me fine-tune Llama 3.2 3B on my customer support dataset.
I have an NVIDIA RTX 4090 with 24GB VRAM.
```

## Agent Configuration

Each agent's `agent.yaml` defines:

```yaml
name: agent-name
version: "1.0.0"
description: What the agent does
modes:
  - mode1
  - mode2
skills:
  - skill_name
requirements:
  - python>=3.9
  - unsloth
```

## Creating New Agents

1. Create a new directory under `agents/`
2. Add `README.md` with documentation
3. Add `agent.yaml` with configuration
4. Implement skills in `skills/`
5. Update this README with the new agent

## Integration with Unsloth

Agents build on top of Unsloth's core APIs:

- `FastLanguageModel` — Model loading and LoRA configuration
- `FastVisionModel` — Vision model support
- `SFTTrainer` — Supervised fine-tuning
- `GRPOTrainer` — Reinforcement learning
- `DPOTrainer` — Direct preference optimization
- Model export utilities (GGUF, merged, Hub)
