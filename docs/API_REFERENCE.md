# API & CLI Reference

This document covers the Unsloth Python API and command-line interface.

## Table of Contents

- [Python API](#python-api)
  - [FastLanguageModel](#fastlanguagemodel)
  - [FastVisionModel](#fastvisionmodel)
  - [Training Utilities](#training-utilities)
  - [Data Preparation](#data-preparation)
  - [Model Registry](#model-registry)
- [CLI Commands](#cli-commands)
  - [train](#unsloth-train)
  - [inference](#unsloth-inference)
  - [export](#unsloth-export)
  - [studio](#unsloth-studio)

## Python API

### FastLanguageModel

The primary interface for loading and fine-tuning language models.

#### `FastLanguageModel.from_pretrained()`

Load a pre-trained model with optional quantization.

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name: str,           # Model name or path (HuggingFace or local)
    max_seq_length: int = 2048,# Maximum sequence length
    dtype: Optional = None,    # Data type (auto-detected if None)
    load_in_4bit: bool = True, # Enable 4-bit quantization
    load_in_8bit: bool = False,# Enable 8-bit quantization
    token: str = None,         # HuggingFace token for gated models
    device_map: str = "auto",  # Device placement strategy
    trust_remote_code: bool = False,  # Allow custom model code
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | Required | HuggingFace model ID or local path |
| `max_seq_length` | int | 2048 | Max tokens per sequence |
| `load_in_4bit` | bool | True | 4-bit NF4 quantization (saves ~75% VRAM) |
| `load_in_8bit` | bool | False | 8-bit quantization |
| `token` | str | None | HuggingFace Hub access token |

**Example:**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)
```

---

#### `FastLanguageModel.get_peft_model()`

Add LoRA adapters to the model for parameter-efficient fine-tuning.

```python
model = FastLanguageModel.get_peft_model(
    model,                          # The loaded model
    r: int = 16,                    # LoRA rank
    target_modules: list = [...],   # Layers to adapt
    lora_alpha: int = 16,           # LoRA scaling factor
    lora_dropout: float = 0,        # Dropout rate
    bias: str = "none",             # Bias training mode
    use_gradient_checkpointing: str = "unsloth",  # Memory optimization
    use_rslora: bool = False,       # Use Rank-Stabilized LoRA
    random_state: int = 3407,       # Random seed
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 16 | LoRA rank (higher = more capacity) |
| `target_modules` | list | All projections | Which layers to adapt |
| `lora_alpha` | int | 16 | Scaling factor (typically equal to `r`) |
| `lora_dropout` | float | 0 | Regularization (0 for SFT, 0.05 for small datasets) |
| `use_gradient_checkpointing` | str | "unsloth" | Memory optimization method |

**Standard target modules:**
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]
```

---

#### Model Saving Methods

```python
# Save LoRA adapters only
model.save_pretrained("path/to/save")
tokenizer.save_pretrained("path/to/save")

# Save merged model (16-bit)
model.save_pretrained_merged("path/to/save", tokenizer, save_method="merged_16bit")

# Save merged model (4-bit)
model.save_pretrained_merged("path/to/save", tokenizer, save_method="merged_4bit_forced")

# Save as GGUF
model.save_pretrained_gguf("path/to/save", tokenizer, quantization_method="q4_k_m")

# Push to HuggingFace Hub
model.push_to_hub("username/model-name")
model.push_to_hub_merged("username/model-name", tokenizer, save_method="merged_16bit")
model.push_to_hub_gguf("username/model-name", tokenizer, quantization_method="q4_k_m")
```

---

#### Inference

```python
# Enable fast inference mode
FastLanguageModel.for_inference(model)

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

### FastVisionModel

Interface for vision-language models.

#### `FastVisionModel.from_pretrained()`

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Llama-3.2-11B-Vision-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

#### `FastVisionModel.get_peft_model()`

```python
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    finetune_vision_layers=True,      # Fine-tune vision encoder
    finetune_language_layers=True,    # Fine-tune language model
    finetune_attention_modules=True,  # Fine-tune attention
    finetune_mlp_modules=True,        # Fine-tune MLP layers
)
```

---

### Training Utilities

#### Chat Templates

```python
from unsloth.chat_templates import get_chat_template

# Apply a model-specific chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",  # Options: llama-3.1, qwen-2.5, gemma, etc.
)

# Format messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

#### DPO Training

```python
from unsloth import PatchDPOTrainer

# Must call before creating DPOTrainer
PatchDPOTrainer()
```

---

### Data Preparation

#### Raw Text Dataset

```python
from unsloth.dataprep.raw_text import create_raw_text_dataset

dataset = create_raw_text_dataset(
    tokenizer=tokenizer,
    text_files=["data/corpus.txt"],
    max_seq_length=2048,
)
```

#### Synthetic Data Generation

```python
from unsloth.dataprep.synthetic import generate_synthetic_data

dataset = generate_synthetic_data(
    config="default",
    num_samples=1000,
)
```

---

### Model Registry

Query supported models and their configurations.

```python
from unsloth.registry import get_supported_models

# List all supported models
models = get_supported_models()

# Check if a model is supported
from unsloth.registry import is_model_supported
supported = is_model_supported("meta-llama/Llama-3.2-3B-Instruct")
```

---

## CLI Commands

The Unsloth CLI is built with Typer and provides commands for training, inference, and export.

### Installation

```bash
pip install unsloth
```

The CLI is available as `unsloth` after installation.

---

### `unsloth train`

Fine-tune a model from the command line.

```bash
unsloth train \
    --model unsloth/Llama-3.2-3B-Instruct \
    --dataset yahma/alpaca-cleaned \
    --output-dir ./outputs \
    --max-seq-length 2048 \
    --load-in-4bit \
    --lora-r 16 \
    --lora-alpha 16 \
    --batch-size 2 \
    --gradient-accumulation 4 \
    --learning-rate 2e-4 \
    --num-epochs 1 \
    --fp16
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model name or path | Required |
| `--dataset` | Dataset name or path | Required |
| `--output-dir` | Output directory | `./outputs` |
| `--max-seq-length` | Max sequence length | 2048 |
| `--load-in-4bit` | Enable 4-bit quantization | False |
| `--lora-r` | LoRA rank | 16 |
| `--lora-alpha` | LoRA alpha | 16 |
| `--batch-size` | Per-device batch size | 2 |
| `--gradient-accumulation` | Gradient accumulation steps | 4 |
| `--learning-rate` | Learning rate | 2e-4 |
| `--num-epochs` | Number of training epochs | 1 |
| `--fp16` | Enable FP16 training | False |

---

### `unsloth inference`

Run inference on a trained model.

```bash
unsloth inference \
    --model ./my-finetuned-model \
    --prompt "Explain quantum computing in simple terms." \
    --max-tokens 256 \
    --temperature 0.7
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model path | Required |
| `--prompt` | Input prompt | Required |
| `--max-tokens` | Maximum output tokens | 256 |
| `--temperature` | Sampling temperature | 0.7 |
| `--top-p` | Top-p sampling | 0.9 |

---

### `unsloth export`

Export a model to various formats.

```bash
# Export to GGUF
unsloth export \
    --model ./my-model \
    --format gguf \
    --quantization q4_k_m \
    --output-dir ./exported

# Export merged model
unsloth export \
    --model ./my-model \
    --format merged \
    --precision 16bit

# Push to HuggingFace Hub
unsloth export \
    --model ./my-model \
    --push-to-hub your-username/model-name
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model path | Required |
| `--format` | Export format (gguf, merged, lora) | gguf |
| `--quantization` | GGUF quantization method | q4_k_m |
| `--precision` | Merge precision (16bit, 4bit) | 16bit |
| `--output-dir` | Output directory | `./exported` |
| `--push-to-hub` | HuggingFace Hub repo ID | None |

---

### `unsloth studio`

Launch the Unsloth Studio web interface.

```bash
unsloth studio \
    --host 0.0.0.0 \
    --port 7860
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--host` | Server host | 0.0.0.0 |
| `--port` | Server port | 7860 |

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | HuggingFace Hub token | For gated models |
| `WANDB_API_KEY` | Weights & Biases key | For experiment tracking |
| `CUDA_VISIBLE_DEVICES` | GPU selection | Optional |
| `UNSLOTH_IS_PRESENT` | Internal flag | Auto-set |

## Next Steps

- [Getting Started](GETTING_STARTED.md) — Quick start guide
- [Fine-Tuning Guide](FINE_TUNING_GUIDE.md) — Detailed training workflows
- [Examples](EXAMPLES.md) — Real-world usage examples
