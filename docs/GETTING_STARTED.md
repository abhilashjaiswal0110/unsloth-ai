# Getting Started with Unsloth

This guide walks you through installing Unsloth and running your first fine-tuning job.

## Prerequisites

- **Python**: 3.9 – 3.14
- **GPU**: NVIDIA (CUDA 11.8+), AMD (ROCm), Intel, or Apple Silicon
- **VRAM**: 6GB+ recommended (4-bit quantization enables smaller GPUs)
- **OS**: Linux, WSL2 (Windows), macOS (Apple Silicon)

## Installation

### Option 1: pip (Recommended)

```bash
# Create a virtual environment
python -m venv unsloth-env
source unsloth-env/bin/activate  # Linux/macOS
# unsloth-env\Scripts\activate   # Windows

# Install Unsloth with HuggingFace support
pip install unsloth[huggingface]
```

### Option 2: From Source (Development)

```bash
git clone https://github.com/abhilashjaiswal0110/unsloth-ai.git
cd unsloth-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[huggingface]"
```

### Option 3: Docker

```bash
docker pull unsloth/unsloth
docker run --gpus all -it unsloth/unsloth
```

### Option 4: Google Colab

Use one of the free Colab notebooks — no local installation needed:

| Model | Notebook |
|-------|----------|
| Llama 3.1 8B | [Open in Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) |
| Qwen 2.5 7B | [Open in Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-GRPO.ipynb) |
| Gemma 3 4B | [Open in Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B)-Conversational.ipynb) |

## Your First Fine-Tune

### Step 1: Load a Pre-trained Model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,        # Use 4-bit quantization to save VRAM
)
```

### Step 2: Add LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                      # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",  # Long context support
)
```

### Step 3: Prepare Your Dataset

```python
from datasets import load_dataset

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Format as chat conversations
def format_prompt(example):
    return {
        "text": f"""### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"""
    }

dataset = dataset.map(format_prompt)
```

### Step 4: Train

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
    ),
)

trainer.train()
```

### Step 5: Save Your Model

```python
# Save LoRA adapters
model.save_pretrained("my-finetuned-model")
tokenizer.save_pretrained("my-finetuned-model")

# Push to HuggingFace Hub
model.push_to_hub("your-username/my-finetuned-model")
```

### Step 6: Run Inference

```python
inputs = tokenizer(
    "### Instruction:\nExplain quantum computing in simple terms.\n\n### Response:\n",
    return_tensors="pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Using the CLI

Unsloth includes a command-line interface for common workflows:

```bash
# Fine-tune a model
unsloth train --model unsloth/Llama-3.2-1B-Instruct --dataset alpaca

# Run inference
unsloth inference --model ./my-finetuned-model --prompt "Hello, how are you?"

# Export model
unsloth export --model ./my-finetuned-model --format gguf --quantization q4_k_m
```

## Next Steps

- [Fine-Tuning Guide](FINE_TUNING_GUIDE.md) — Deep dive into fine-tuning options
- [Data Preparation](DATA_PREPARATION.md) — Prepare custom datasets
- [Examples](EXAMPLES.md) — Real-world use cases and prompts
- [RL Training Guide](RL_TRAINING_GUIDE.md) — Reinforcement learning with GRPO
