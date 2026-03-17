# Unsloth Documentation

Welcome to the Unsloth documentation. Unsloth provides **2-5X faster training, reinforcement learning & fine-tuning** of large language models with up to 70% less memory usage.

## Quick Navigation

| Document | Description | Audience |
|----------|-------------|----------|
| [Getting Started](GETTING_STARTED.md) | Installation & first fine-tune | New users |
| [Local Development](LOCAL_DEVELOPMENT.md) | Dev environment setup | Contributors |
| [Fine-Tuning Guide](FINE_TUNING_GUIDE.md) | Complete fine-tuning workflows | ML engineers |
| [RL Training Guide](RL_TRAINING_GUIDE.md) | GRPO & reinforcement learning | ML researchers |
| [Data Preparation](DATA_PREPARATION.md) | Dataset formatting & prep | Data engineers |
| [Model Export Guide](MODEL_EXPORT_GUIDE.md) | Export to GGUF, vLLM, etc. | MLOps engineers |
| [API Reference](API_REFERENCE.md) | CLI commands & Python API | Developers |
| [Examples](EXAMPLES.md) | Prompts & use case examples | All users |
| [Troubleshooting](TROUBLESHOOTING.md) | Common issues & solutions | All users |

## Supported Models

Unsloth supports 2-5x faster fine-tuning for these model families:

- **Qwen** — Qwen 2, 2.5, 3, 3.5 (including MoE variants)
- **Llama** — Llama 2, 3.1, 3.2, 3.3, 4
- **Gemma** — Gemma 2, 3
- **Mistral** — Mistral, Ministral
- **DeepSeek** — DeepSeek V2, V3, R1
- **Phi** — Phi-3, Phi-4
- **And more** — Cohere, Falcon, Granite, GLM

## Architecture Overview

```
unsloth/
├── models/          # Model implementations & LoRA patching
├── kernels/         # Custom Triton/CUDA kernels for speed
├── dataprep/        # Data preprocessing & synthetic generation
├── registry/        # Model auto-discovery & registration
├── utils/           # Attention, packing, HuggingFace Hub utilities
├── trainer.py       # Training orchestration
├── save.py          # Model serialization (GGUF, safetensors, etc.)
└── chat_templates.py # Chat template mappings for all models
```

## Getting Help

- **Documentation**: You're here!
- **Discord**: [discord.gg/unsloth](https://discord.gg/unsloth)
- **Reddit**: [r/unsloth](https://reddit.com/r/unsloth)
- **GitHub Issues**: [Report bugs](https://github.com/abhilashjaiswal0110/unsloth-ai/issues)
