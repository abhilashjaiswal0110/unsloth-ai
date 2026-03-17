# Model Export Guide

This guide covers exporting fine-tuned models to various formats for deployment.

## Table of Contents

- [Export Formats](#export-formats)
- [GGUF Export (Ollama / llama.cpp)](#gguf-export-ollama--llamacpp)
- [vLLM Export](#vllm-export)
- [HuggingFace Hub](#huggingface-hub)
- [Merged Model Export](#merged-model-export)
- [Quantization Options](#quantization-options)
- [Deployment Patterns](#deployment-patterns)

## Export Formats

| Format | Platform | Use Case |
|--------|----------|----------|
| **GGUF** | Ollama, llama.cpp | Local inference, edge devices |
| **SafeTensors** | HuggingFace, vLLM | Cloud inference, API serving |
| **LoRA Adapters** | Any HF-compatible | Lightweight, stackable adapters |
| **Merged 16-bit** | Any platform | Full precision deployment |
| **Merged 4-bit** | HF, vLLM | Memory-efficient deployment |

## GGUF Export (Ollama / llama.cpp)

### Export to GGUF

```python
from unsloth import FastLanguageModel

# Load your fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./my-finetuned-model",
    max_seq_length=2048,
)

# Export to GGUF with quantization
model.save_pretrained_gguf(
    "my-model-gguf",
    tokenizer,
    quantization_method="q4_k_m",  # 4-bit quantization
)
```

### GGUF Quantization Methods

| Method | Size | Quality | Speed | Use Case |
|--------|------|---------|-------|----------|
| `q2_k` | Smallest | Lower | Fastest | Edge devices, mobile |
| `q4_0` | Small | Good | Fast | General local inference |
| `q4_k_m` | Small | Better | Fast | **Recommended default** |
| `q5_k_m` | Medium | High | Medium | Quality-focused |
| `q8_0` | Large | Highest | Slower | Maximum quality |
| `f16` | Largest | Lossless | Slowest | Reference, benchmarking |

### Use with Ollama

```bash
# After GGUF export, create Modelfile
cat > Modelfile << 'EOF'
FROM ./my-model-gguf/unsloth.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
EOF

# Create Ollama model
ollama create my-model -f Modelfile

# Run it
ollama run my-model "Hello, how are you?"
```

### Use with llama.cpp

```bash
# Direct inference
./llama-cli -m my-model-gguf/unsloth.Q4_K_M.gguf \
    -p "### Instruction:\nExplain quantum computing.\n\n### Response:" \
    -n 256 --temp 0.7
```

## vLLM Export

### Export for vLLM Serving

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./my-finetuned-model",
    max_seq_length=2048,
)

# Save as merged 16-bit model for vLLM
model.save_pretrained_merged(
    "my-model-vllm",
    tokenizer,
    save_method="merged_16bit",
)
```

### Deploy with vLLM

```bash
# Install vLLM
pip install vllm

# Start API server
python -m vllm.entrypoints.openai.api_server \
    --model ./my-model-vllm \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 2048
```

### Query the API

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "./my-model-vllm",
        "prompt": "Explain machine learning in simple terms.",
        "max_tokens": 256,
        "temperature": 0.7,
    },
)
print(response.json()["choices"][0]["text"])
```

## HuggingFace Hub

### Push LoRA Adapters

```python
# Save and push just the LoRA adapters (smallest upload)
model.save_pretrained("my-model-lora")
tokenizer.save_pretrained("my-model-lora")

model.push_to_hub("your-username/my-model-lora")
tokenizer.push_to_hub("your-username/my-model-lora")
```

### Push Merged Model

```python
# Push full merged model
model.push_to_hub_merged(
    "your-username/my-model-merged",
    tokenizer,
    save_method="merged_16bit",
)
```

### Push GGUF to Hub

```python
model.push_to_hub_gguf(
    "your-username/my-model-gguf",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0"],  # Multiple formats
)
```

## Merged Model Export

### 16-bit Merge

```python
# Full precision merge — largest but highest quality
model.save_pretrained_merged(
    "my-model-16bit",
    tokenizer,
    save_method="merged_16bit",
)
```

### 4-bit Merge

```python
# Quantized merge — smaller file size
model.save_pretrained_merged(
    "my-model-4bit",
    tokenizer,
    save_method="merged_4bit_forced",
)
```

### LoRA Only

```python
# Just save the adapter weights (smallest)
model.save_pretrained("my-model-lora-only")
tokenizer.save_pretrained("my-model-lora-only")
```

## Quantization Options

### Choosing Quantization

```
Quality:    f16 > q8_0 > q5_k_m > q4_k_m > q4_0 > q2_k
Size:       q2_k < q4_0 < q4_k_m < q5_k_m < q8_0 < f16
Speed:      q4_0 > q4_k_m > q2_k > q5_k_m > q8_0 > f16
```

### Size Estimates (7B Parameter Model)

| Quantization | Size | VRAM |
|-------------|------|------|
| f16 | ~14 GB | ~16 GB |
| q8_0 | ~7 GB | ~9 GB |
| q5_k_m | ~5 GB | ~7 GB |
| q4_k_m | ~4 GB | ~6 GB |
| q4_0 | ~3.8 GB | ~5.5 GB |
| q2_k | ~2.5 GB | ~4 GB |

## Deployment Patterns

### Pattern 1: Local Development (Ollama)

```
Train → GGUF q4_k_m → Ollama → localhost API
```

Best for: Personal use, prototyping, development

### Pattern 2: Production API (vLLM)

```
Train → Merged 16-bit → vLLM → OpenAI-compatible API
```

Best for: Production APIs, high-throughput serving

### Pattern 3: Hub Distribution

```
Train → Push LoRA + GGUF → HuggingFace Hub → Community access
```

Best for: Open-source models, community sharing

### Pattern 4: Edge Deployment

```
Train → GGUF q2_k → llama.cpp → Mobile / embedded device
```

Best for: Mobile apps, IoT devices, offline usage

## CLI Export

```bash
# Export to GGUF
unsloth export --model ./my-model --format gguf --quantization q4_k_m

# Export merged model
unsloth export --model ./my-model --format merged --precision 16bit

# Push to Hub
unsloth export --model ./my-model --push-to-hub your-username/model-name
```

## Troubleshooting

### "Model too large for GPU"
- Use a smaller quantization method (q4_0 or q2_k)
- Use CPU inference with llama.cpp

### "GGUF export fails"
- Ensure you have the latest `llama-cpp-python` installed
- Check that the model architecture is supported

### "vLLM compatibility issues"
- Use `merged_16bit` save method for best vLLM compatibility
- Verify vLLM version supports your model architecture

## Next Steps

- [Fine-Tuning Guide](FINE_TUNING_GUIDE.md) — Train your model before exporting
- [Troubleshooting](TROUBLESHOOTING.md) — Common export issues
- [Examples](EXAMPLES.md) — End-to-end export examples
