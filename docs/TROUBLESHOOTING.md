# Troubleshooting Guide

This guide covers common issues and solutions when using Unsloth.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Model Loading Issues](#model-loading-issues)
- [Training Issues](#training-issues)
- [Memory Issues](#memory-issues)
- [Export Issues](#export-issues)
- [Performance Issues](#performance-issues)
- [Platform-Specific Issues](#platform-specific-issues)
- [Getting Help](#getting-help)

## Installation Issues

### "No module named 'unsloth'"

**Cause**: Unsloth is not installed in your active Python environment.

**Solution**:
```bash
# Verify your virtual environment is active
which python  # Should show your venv path

# Install Unsloth
pip install unsloth

# Or with HuggingFace support
pip install unsloth[huggingface]
```

### "CUDA not available"

**Cause**: PyTorch is installed without CUDA support.

**Solution**:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

### "Triton not found" (Linux only)

**Cause**: Triton kernels are not installed.

**Solution**:
```bash
pip install unsloth[triton]
# or
pip install triton
```

### Dependency conflicts

**Cause**: Version mismatches between transformers, trl, peft, etc.

**Solution**:
```bash
# Create a clean environment
python -m venv clean-env
source clean-env/bin/activate
pip install unsloth[huggingface]
```

---

## Model Loading Issues

### "Model not found"

**Cause**: Incorrect model name or missing authentication.

**Solution**:
```python
# Use the correct model name format
# ✅ Correct
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Llama-3.2-3B-Instruct")

# ❌ Wrong
model, tokenizer = FastLanguageModel.from_pretrained("llama-3.2-3b")

# For gated models, provide HF token
model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    token="hf_your_token_here",
)
```

### "Out of memory during model loading"

**Cause**: Model is too large for available VRAM.

**Solution**:
```python
# Use 4-bit quantization (saves ~75% VRAM)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    load_in_4bit=True,
    max_seq_length=2048,
)

# Or use a smaller model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",  # 1B instead of 3B
    load_in_4bit=True,
)
```

### "Unsupported model architecture"

**Cause**: The model architecture is not yet supported by Unsloth.

**Solution**:
```python
# Check supported models
from unsloth.registry import get_supported_models
models = get_supported_models()
print(models)

# Use a supported variant (Unsloth-optimized versions)
# Check: https://huggingface.co/unsloth
```

---

## Training Issues

### "Loss is NaN or exploding"

**Cause**: Learning rate is too high or data has issues.

**Solution**:
```python
args = TrainingArguments(
    learning_rate=1e-5,            # Lower learning rate
    warmup_steps=10,               # Add warmup
    fp16=True,                     # Use FP16 (not BF16 on older GPUs)
    max_grad_norm=1.0,             # Gradient clipping
    gradient_accumulation_steps=8, # Larger effective batch
)
```

### "Training is very slow"

**Cause**: Inefficient configuration.

**Solution**:
```python
# Enable packing for short sequences
trainer = SFTTrainer(
    ...
    packing=True,                  # Pack multiple samples per batch
)

# Use gradient checkpointing
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
)

# Increase batch size if VRAM allows
args = TrainingArguments(
    per_device_train_batch_size=4,  # Increase from 2
    gradient_accumulation_steps=4,
)
```

### "Dataset format error"

**Cause**: Dataset doesn't match expected format.

**Solution**:
```python
# Verify dataset columns
print(dataset.column_names)
print(dataset[0])

# Ensure 'text' field exists for SFTTrainer
def format_data(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"}

dataset = dataset.map(format_data)
```

### "Tokenizer warning: sequence too long"

**Cause**: Input sequences exceed `max_seq_length`.

**Solution**:
```python
# Increase max_seq_length
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="...",
    max_seq_length=4096,  # Increase from default 2048
)

# Or filter long sequences
def filter_length(example):
    tokens = tokenizer(example["text"], truncation=False)
    return len(tokens["input_ids"]) <= 2048

dataset = dataset.filter(filter_length)
```

---

## Memory Issues

### "CUDA out of memory"

**Cause**: GPU VRAM exceeded during training or inference.

**Solutions** (try in order):

```python
# 1. Reduce batch size
args = TrainingArguments(per_device_train_batch_size=1)

# 2. Enable gradient checkpointing
model = FastLanguageModel.get_peft_model(
    model, use_gradient_checkpointing="unsloth"
)

# 3. Use 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="...", load_in_4bit=True
)

# 4. Reduce sequence length
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="...", max_seq_length=1024  # Shorter sequences
)

# 5. Lower LoRA rank
model = FastLanguageModel.get_peft_model(model, r=8)  # Lower rank

# 6. Clear GPU cache
import torch
torch.cuda.empty_cache()
```

### Monitor VRAM Usage

```python
import torch

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Total:     {total:.2f} GB")
    print(f"Free:      {total - allocated:.2f} GB")

print_gpu_memory()
```

### VRAM Requirements

| Model Size | 4-bit | 8-bit | 16-bit |
|-----------|-------|-------|--------|
| 1B | ~2 GB | ~3 GB | ~4 GB |
| 3B | ~4 GB | ~6 GB | ~8 GB |
| 7B | ~6 GB | ~10 GB | ~16 GB |
| 13B | ~10 GB | ~16 GB | ~28 GB |
| 70B | ~42 GB | ~72 GB | ~140 GB |

*Training requires roughly 2x the inference memory*

---

## Export Issues

### "GGUF export fails"

**Cause**: Missing llama.cpp dependencies or unsupported architecture.

**Solution**:
```bash
# Install/update llama-cpp-python
pip install llama-cpp-python --upgrade

# If still failing, try merging first then converting
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('./my-model')
model.save_pretrained_merged('merged-model', tokenizer, save_method='merged_16bit')
"
```

### "Push to Hub fails"

**Cause**: Authentication or repository issues.

**Solution**:
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token in code
model.push_to_hub("username/model", token="hf_your_token")
```

---

## Performance Issues

### Slow training compared to benchmarks

**Checklist**:

1. **Triton installed?** `pip install triton` (Linux only)
2. **4-bit quantization enabled?** `load_in_4bit=True`
3. **Gradient checkpointing?** `use_gradient_checkpointing="unsloth"`
4. **Packing enabled?** `packing=True` in SFTTrainer
5. **Correct batch size?** Maximize within VRAM limits
6. **Not CPU-bound?** Check if data loading is the bottleneck

### Expected Training Speeds

| Model | GPU | Speed (tokens/sec) |
|-------|-----|-------------------|
| Llama 3.2 1B | RTX 3090 | ~8,000 |
| Llama 3.2 3B | RTX 3090 | ~4,500 |
| Llama 3.1 8B | A100 40GB | ~6,000 |
| Llama 3.1 70B | 4x A100 80GB | ~2,000 |

---

## Platform-Specific Issues

### Windows (WSL2)

```bash
# Ensure WSL2 is updated
wsl --update

# Install NVIDIA drivers in Windows (not WSL)
# Then in WSL:
nvidia-smi  # Should show your GPU
```

### macOS (Apple Silicon)

```bash
# Use MPS backend
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Install PyTorch for Apple Silicon
pip install torch torchvision torchaudio
```

### AMD GPUs

```bash
# Set ROCm device
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Adjust for your GPU

# Install PyTorch for ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

---

## Getting Help

### Before Asking for Help

1. **Check this guide** for your specific error
2. **Search existing issues**: [GitHub Issues](https://github.com/abhilashjaiswal0110/unsloth-ai/issues)
3. **Include diagnostic info**:

```python
import torch
import sys
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

import unsloth
print(f"Unsloth: {unsloth.__version__}")
```

### Where to Get Help

| Channel | Best For |
|---------|----------|
| [GitHub Issues](https://github.com/abhilashjaiswal0110/unsloth-ai/issues) | Bug reports, feature requests |
| [Discord](https://discord.gg/unsloth) | Quick questions, community help |
| [Reddit r/unsloth](https://reddit.com/r/unsloth) | Discussions, showcase models |

### Reporting a Bug

When filing an issue, include:

1. **What you expected** to happen
2. **What actually happened** (full error traceback)
3. **Steps to reproduce** (minimal code example)
4. **Environment info** (OS, Python, PyTorch, CUDA, GPU)
5. **Model and dataset** used
