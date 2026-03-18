# Windows Setup — Qwen3 Advanced GRPO Training

Complete guide for running **Qwen3** fine-tuning with **Advanced GRPO** (Group Relative Policy Optimization) on **Windows**, using a fully isolated conda environment that does not interfere with any existing Python or PyTorch installations.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Quick Start (5 minutes)](#quick-start)
- [Installation — Step by Step](#installation)
- [Configuration Reference](#configuration-reference)
- [Running GRPO Training](#running-grpo-training)
- [Reward Function Design](#reward-function-design)
- [Model Size & VRAM Guide](#model-size--vram-guide)
- [Monitoring Training](#monitoring-training)
- [Saving & Exporting the Model](#saving--exporting-the-model)
- [Troubleshooting (Windows-specific)](#troubleshooting)

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows 10 (22H2) | Windows 11 |
| GPU | NVIDIA, 6 GB VRAM (4-bit) | RTX 3080+ / 16 GB VRAM |
| CUDA Driver | 520+ (for CUDA 11.8) | 550+ (for CUDA 12.4) |
| Disk Space | 20 GB | 50 GB |
| RAM | 16 GB | 32 GB |
| Internet | Required for model download | — |

> **Note**: Flash Attention is not available as a pre-built wheel on Windows.
> Unsloth uses **xformers** attention + custom Triton kernels (`triton-windows`) instead,
> achieving comparable speed on NVIDIA GPUs.

### Check your GPU driver and CUDA version

```powershell
nvidia-smi
```

Look for the **CUDA Version** in the top-right corner of the output (e.g., `12.4`).

---

## Architecture Overview

```
unsloth-ai/
├── setup/windows/
│   ├── install.ps1        ← Main installer (creates isolated conda env)
│   ├── install.bat        ← Double-click wrapper for install.ps1
│   └── environment.yml    ← Conda environment specification
├── configs/
│   └── qwen3_grpo.yaml    ← Training hyperparameters (edit this)
├── scripts/
│   └── train_qwen3_grpo.py ← Training entry point
└── docs/
    └── WINDOWS_QWEN3_GRPO_SETUP.md  ← This file
```

**Isolated environment**: everything installs into a conda env named
`unsloth-qwen3-grpo` — no changes to your system Python or existing envs.

---

## Quick Start

```powershell
# 1. Open PowerShell (or Windows Terminal) in the repo root
cd "C:\path\to\unsloth-ai"

# 2. Run the installer (installs Miniconda if needed, creates isolated env)
setup\windows\install.bat

# 3. Activate the environment
conda activate unsloth-qwen3-grpo

# 4. Add your HuggingFace token to .env
#    (Edit .env that was created by the installer)
notepad .env

# 5. Start training
python scripts\train_qwen3_grpo.py --config configs\qwen3_grpo.yaml
```

---

## Installation

### Option A: One-command installer (recommended)

```powershell
# From repository root — runs in user context, no admin required
setup\windows\install.bat
```

Or run the PowerShell script directly with optional parameters:

```powershell
# Default: CUDA 12.4, Python 3.11
powershell -ExecutionPolicy Bypass -File setup\windows\install.ps1

# CUDA 12.1 on Python 3.12
powershell -ExecutionPolicy Bypass -File setup\windows\install.ps1 `
    -CudaVersion 12.1 -PythonVersion 3.12

# Force recreate the environment from scratch
powershell -ExecutionPolicy Bypass -File setup\windows\install.ps1 -Reinstall
```

The installer performs these steps automatically:

1. Detects NVIDIA GPU and driver version
2. Installs Miniconda3 (user-only) if conda is not found
3. Creates the `unsloth-qwen3-grpo` conda environment
4. Installs PyTorch with the specified CUDA version
5. Installs `triton-windows`, `bitsandbytes`, `xformers` (Windows-specific)
6. Installs the HuggingFace ecosystem (transformers, peft, trl, accelerate, datasets)
7. Installs Unsloth from source in editable mode (`pip install -e . --no-deps`)
8. Writes a `.env` template with credential placeholders
9. Runs a smoke-test to verify the entire stack

### Option B: Manual conda environment

```powershell
# Create from spec file (exact pinned versions)
conda env create -f setup\windows\environment.yml
conda activate unsloth-qwen3-grpo

# Install PyTorch manually (adjust cudaXXX to match your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Unsloth from source
pip install -e . --no-deps
```

### Verify installation

```powershell
conda activate unsloth-qwen3-grpo
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import unsloth; print(unsloth.__version__)"
```

---

## Configuration Reference

Edit [`configs/qwen3_grpo.yaml`](../configs/qwen3_grpo.yaml) to customise training.

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `unsloth/Qwen3-8B` | Model to fine-tune (see [VRAM guide](#model-size--vram-guide)) |
| `max_seq_length` | `4096` | Token context window |
| `load_in_4bit` | `true` | 4-bit NF4 quantization (saves 70% VRAM) |
| `lora_rank` | `32` | LoRA rank — higher = more capacity, more VRAM |
| `num_generations` | `6` | Responses per prompt (GRPO rollout size) |
| `max_completion_length` | `1024` | Max tokens generated per rollout |
| `learning_rate` | `5e-6` | Keep low for RL (1e-6 – 1e-5 range) |
| `bf16` | `true` | Use bfloat16 (Ampere/Ada GPUs); set `fp16: true` for older |
| `dataset` | `openai/gsm8k` | HF dataset ID |
| `weight_correctness` | `0.40` | Reward weight for numerical answer accuracy |
| `weight_format` | `0.30` | Reward weight for `<think>` + `\boxed{}` format |
| `weight_reasoning` | `0.20` | Reward weight for structured reasoning chains |
| `weight_length` | `0.10` | Reward weight for ideal response length |

### Using a custom dataset

Your dataset needs at minimum:
- `question` (or `prompt`) — the input prompt
- `answer` (or `solution`) — the ground-truth answer

```yaml
dataset: your-org/your-dataset
dataset_config: ""         # leave blank if no config name
dataset_split: train
dataset_eval_split: test
```

For datasets that don't have `####`-delimited answers, override `_build_prompt`
in `scripts/train_qwen3_grpo.py` to extract the numeric answer your way.

---

## Running GRPO Training

### Basic run

```powershell
conda activate unsloth-qwen3-grpo
python scripts\train_qwen3_grpo.py --config configs\qwen3_grpo.yaml
```

### CLI overrides

All YAML keys can be overridden from the command line:

```powershell
# Quick experiment: cap at 100 steps with smaller model
python scripts\train_qwen3_grpo.py ^
    --config configs\qwen3_grpo.yaml ^
    --model unsloth/Qwen3-4B ^
    --max-steps 100 ^
    --report-to none

# Full run with W&B logging
python scripts\train_qwen3_grpo.py ^
    --config configs\qwen3_grpo.yaml ^
    --run-name qwen3-8b-gsm8k-v1 ^
    --report-to wandb
```

### Low-VRAM mode (< 12 GB)

```powershell
python scripts\train_qwen3_grpo.py ^
    --config configs\qwen3_grpo.yaml ^
    --model unsloth/Qwen3-4B ^
    --lora-rank 16 ^
    --max-steps 200
```

In the YAML, also reduce:

```yaml
num_generations: 4          # from 6
max_completion_length: 512  # from 1024
gradient_accumulation_steps: 8
```

---

## Reward Function Design

The training script uses a **four-component composite reward**:

```
total_reward = w_fmt  × format_reward
             + w_corr × correctness_reward
             + w_reas × reasoning_reward
             + w_len  × length_reward
```

### format_reward
Rewards the model for using Qwen3's thinking mode correctly:
- +0.5 if `<think>…</think>` block is present
- +0.2 if the thinking block has ≥ 10 words (non-trivial)
- +0.3 if the answer is inside `\boxed{}`

### correctness_reward
Numerically compares the extracted `\boxed{}` value against the ground-truth.
- 1.0 for exact or numerically close match
- 0.3 partial credit if the answer appears in the response text

### reasoning_reward
Rewards structured reasoning inside `<think>`:
- Up to 0.4 for explicit step markers (`Step 1:`, numbered lists)
- Up to 0.2 for equations (`=` signs)
- Up to 0.2 for logical connectives (`therefore`, `because`, `hence`, …)

### length_reward
Gaussian-shaped reward centred at ~350 words — penalises both
trivially short responses and unnecessarily verbose ones.

### Customising rewards

To add a custom reward (e.g., JSON format verification):

```python
# In scripts/train_qwen3_grpo.py, add your function:
def json_format_reward(completions: list[str], **kwargs) -> list[float]:
    import json
    scores = []
    for c in completions:
        try:
            json.loads(c); scores.append(1.0)
        except json.JSONDecodeError:
            scores.append(-0.5)
    return scores
```

Then pass it as an additional reward function in `train()`:

```python
trainer = GRPOTrainer(
    reward_funcs=[reward_fn, json_format_reward],
    ...
)
```

---

## Model Size & VRAM Guide

| Model | VRAM (4-bit) | VRAM (16-bit) | Recommended GPU |
|-------|-------------|--------------|-----------------|
| `unsloth/Qwen3-1.7B` | ~4 GB | ~4 GB | RTX 3060, GTX 1080Ti |
| `unsloth/Qwen3-4B` | ~6 GB | ~9 GB | RTX 3070, RTX 4060 |
| `unsloth/Qwen3-8B` | ~10 GB | ~17 GB | RTX 3080, RTX 4070 |
| `unsloth/Qwen3-14B` | ~16 GB | ~30 GB | RTX 4090, A100-40G |
| `unsloth/Qwen3-32B` | ~24 GB | ~68 GB | A100-80G, H100 |

> Unsloth's gradient checkpointing (`gradient_checkpointing: unsloth`) can reduce
> activation VRAM by ~30% at the cost of ~5% throughput reduction.

### Enable thinking mode

Qwen3's extended thinking is enabled via `enable_thinking=True` in the chat template call.
The training script sets this automatically. At inference time:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("outputs/qwen3-grpo")
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "What is 15% of 200?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

---

## Monitoring Training

### Weights & Biases

```powershell
# Set your key in .env (already present after install)
# WANDB_API_KEY=your_key_here

# Or login interactively
conda activate unsloth-qwen3-grpo
wandb login
```

The run will appear at `https://wandb.ai/<entity>/<project>`.

Key metrics to watch:
- `train/reward` — should increase steadily
- `train/kl` — KL divergence from reference policy; keep < 0.5
- `train/loss` — composite policy loss
- `eval/reward` — held-out prompt reward (main quality signal)

### TensorBoard

```powershell
# Start TensorBoard
conda activate unsloth-qwen3-grpo
tensorboard --logdir outputs/qwen3-grpo

# Navigate to http://localhost:6006
```

---

## Saving & Exporting the Model

### Save LoRA adapters (default, smallest)

```python
model.save_pretrained("outputs/qwen3-grpo")
tokenizer.save_pretrained("outputs/qwen3-grpo")
```

### Merge and save in 16-bit (standalone model)

```python
model.save_pretrained_merged("outputs/qwen3-grpo-merged", tokenizer, save_method="merged_16bit")
```

### Export to GGUF (for llama.cpp / Ollama)

```python
model.save_pretrained_gguf("outputs/qwen3-grpo-gguf", tokenizer, quantization_method="q4_k_m")
```

See [MODEL_EXPORT_GUIDE.md](MODEL_EXPORT_GUIDE.md) for full export options.

---

## Troubleshooting

### `bitsandbytes` CUDA errors on Windows

```
CUDA error: no kernel image is available for execution on the device
```

Ensure the bitsandbytes version matches your PyTorch CUDA version:

```powershell
conda activate unsloth-qwen3-grpo
pip install bitsandbytes --upgrade
```

If the issue persists, try the `bitsandbytes-windows` fork:

```powershell
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

### `triton` import errors

`triton` (Linux) will not install on Windows; `triton-windows` replaces it.
Verify:

```powershell
conda activate unsloth-qwen3-grpo
python -c "import triton; print(triton.__version__)"
```

If missing: `pip install triton-windows`

### Out of VRAM (OOM) errors

1. **Reduce `num_generations`** from 6 to 4 or 2.
2. **Reduce `max_completion_length`** from 1024 to 512.
3. **Increase `gradient_accumulation_steps`** and reduce `per_device_train_batch_size` to 1.
4. **Switch to a smaller model** (e.g., `Qwen3-4B` instead of `Qwen3-8B`).
5. **Enable disk offloading**: set `UNSLOTH_OFFLOAD_TO_DISK=1` in `.env`.

```powershell
# Quick OOM diagnostic
python -c "import torch; print(torch.cuda.memory_summary())"
```

### Slow downloads / HuggingFace timeout

Enable `hf_transfer` for 5-10× faster model downloads:

```bash
# In .env
HF_HUB_ENABLE_HF_TRANSFER=1
```

Or set a local cache directory on a fast drive:

```powershell
$env:HF_HOME = "D:\hf_cache"     # PowerShell
```

### `conda activate` not working in PowerShell

Run conda init first:

```powershell
conda init powershell
# Then restart PowerShell and try again
conda activate unsloth-qwen3-grpo
```

### Environment conflicts with existing Python

The installer creates a **completely isolated** conda environment.
It does **not** touch system Python, pip, or any other conda env.
If you still see conflicts, check that the right Python is active:

```powershell
conda activate unsloth-qwen3-grpo
where python          # Should point to miniconda3\envs\unsloth-qwen3-grpo\python.exe
python --version      # Should be 3.11.x
```

---

## Next Steps

- [RL Training Guide](RL_TRAINING_GUIDE.md) — Detailed GRPO theory and variants
- [Fine-Tuning Guide](FINE_TUNING_GUIDE.md) — SFT before RL
- [Model Export Guide](MODEL_EXPORT_GUIDE.md) — GGUF, vLLM, HuggingFace Hub
- [Troubleshooting](TROUBLESHOOTING.md) — General Unsloth issues
