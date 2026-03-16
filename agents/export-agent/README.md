# Export Agent

An agent for exporting fine-tuned models to production-ready formats for deployment.

## Overview

The Export Agent handles model serialization and deployment:

1. **Format Selection** — Choose the right export format for your use case
2. **Quantization** — Apply optimal quantization for size/quality trade-off
3. **Validation** — Verify exported model works correctly
4. **Deployment Setup** — Generate deployment configurations (Ollama, vLLM, etc.)

## Modes

| Mode | Description |
|------|-------------|
| `gguf` | Export to GGUF for Ollama / llama.cpp |
| `vllm` | Export for vLLM serving |
| `hub` | Push to HuggingFace Hub |
| `merged` | Export merged full-weight model |

## Usage

### Export to GGUF with Ollama Setup

```python
from agents.export_agent.skills.export_model import export_to_gguf

result = export_to_gguf(
    model_path="./my-finetuned-model",
    output_dir="./exported",
    quantization="q4_k_m",
    generate_modelfile=True,
)

print(f"Exported to: {result['output_path']}")
print(f"Modelfile: {result['modelfile_path']}")
```

### Export for vLLM

```python
from agents.export_agent.skills.export_model import export_for_vllm

result = export_for_vllm(
    model_path="./my-finetuned-model",
    output_dir="./vllm-model",
)
```

### AI Assistant Prompt

```
I've fine-tuned a Qwen 2.5 7B model for code generation. Help me:
1. Export it as GGUF q4_k_m for local Ollama usage
2. Create an Ollama Modelfile with the right chat template
3. Also push the LoRA adapters to HuggingFace Hub
4. Generate a vLLM deployment config for production
```

## Skills

| Skill | Purpose |
|-------|---------|
| `export_model` | Export models to GGUF, merged, or Hub formats |
| `generate_deployment` | Generate Ollama Modelfile or vLLM configs |
| `validate_export` | Verify exported model loads and generates correctly |
