# Fine-Tuning Guide

This guide covers all fine-tuning workflows supported by Unsloth, with detailed use cases, configurations, and prompts.

## Table of Contents

- [Overview](#overview)
- [Use Case 1: Instruction Following](#use-case-1-instruction-following)
- [Use Case 2: Conversational Chatbot](#use-case-2-conversational-chatbot)
- [Use Case 3: Code Generation](#use-case-3-code-generation)
- [Use Case 4: Domain-Specific Knowledge](#use-case-4-domain-specific-knowledge)
- [Use Case 5: Vision Model Fine-Tuning](#use-case-5-vision-model-fine-tuning)
- [Use Case 6: Embedding Model Fine-Tuning](#use-case-6-embedding-model-fine-tuning)
- [Advanced Configuration](#advanced-configuration)
- [Model Selection Guide](#model-selection-guide)

## Overview

Unsloth accelerates fine-tuning by 2-5x through custom Triton kernels and memory optimizations. It supports:

- **SFT (Supervised Fine-Tuning)** — Train on instruction-response pairs
- **LoRA / QLoRA** — Parameter-efficient fine-tuning with 4-bit quantization
- **DPO** — Direct Preference Optimization for alignment
- **GRPO** — Group Relative Policy Optimization (see [RL Training Guide](RL_TRAINING_GUIDE.md))
- **Vision** — Fine-tune multimodal vision-language models
- **Embedding** — Fine-tune sentence transformer models

## Use Case 1: Instruction Following

**Goal**: Train a model to follow specific instructions accurately.

### Dataset Format

```json
{
  "instruction": "Summarize the following text in 3 bullet points.",
  "input": "Artificial intelligence has transformed...",
  "output": "• AI has revolutionized...\n• Key applications include...\n• Future developments..."
}
```

### Training Script

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# Load and format dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

alpaca_prompt = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

def format_prompts(examples):
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(alpaca_prompt.format(
            instruction=instruction, input=inp, output=output
        ) + tokenizer.eos_token)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,  # Pack short sequences for efficiency
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs/instruction-model",
        seed=42,
    ),
)

trainer.train()
model.save_pretrained("instruction-model-lora")
```

### Example Prompts for Testing

```
### Instruction:
Write a Python function that calculates the Fibonacci sequence up to n terms.

### Response:
```

```
### Instruction:
Explain the theory of relativity to a 10-year-old.

### Response:
```

## Use Case 2: Conversational Chatbot

**Goal**: Create a conversational assistant with chat-style interactions.

### Dataset Format (ShareGPT)

```json
{
  "conversations": [
    {"from": "human", "value": "What is machine learning?"},
    {"from": "gpt", "value": "Machine learning is a subset of AI..."},
    {"from": "human", "value": "How does it differ from deep learning?"},
    {"from": "gpt", "value": "Deep learning is a specialized form..."}
  ]
}
```

### Training Script

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# Apply chat template
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# Load conversation dataset
dataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train")

def format_conversations(examples):
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in examples["conversations"]
    ]
    return {"text": texts}

dataset = dataset.map(format_conversations, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs/chatbot-model",
    ),
)

trainer.train()
```

### Example Prompts

```
User: I'm planning a trip to Japan. What should I know before going?
Assistant:
```

```
User: Can you help me debug this Python error: "TypeError: 'NoneType' object is not iterable"?
Assistant:
```

## Use Case 3: Code Generation

**Goal**: Fine-tune a model to generate high-quality code.

### Training Script

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=8192,    # Longer context for code
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,                   # Higher rank for code tasks
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

# Load code dataset
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

code_prompt = """### Task:
{instruction}

### Input:
{input}

### Solution:
{output}"""

def format_code(examples):
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(code_prompt.format(
            instruction=instruction,
            input=inp or "",
            output=output
        ) + tokenizer.eos_token)
    return {"text": texts}

dataset = dataset.map(format_code, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=8192,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs/code-model",
    ),
)

trainer.train()
```

### Example Prompts

```
### Task:
Write a REST API endpoint in FastAPI that handles user authentication with JWT tokens.

### Solution:
```

```
### Task:
Create a Python class that implements a thread-safe LRU cache with TTL support.

### Solution:
```

## Use Case 4: Domain-Specific Knowledge

**Goal**: Fine-tune on domain-specific data (medical, legal, financial, etc.).

### Dataset Format

```json
{
  "question": "What are the symptoms of Type 2 diabetes?",
  "context": "Type 2 diabetes is a chronic condition...",
  "answer": "Common symptoms include increased thirst, frequent urination..."
}
```

### Training Script

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Gemma-3-4B-it",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,    # Small dropout for domain-specific data
    use_gradient_checkpointing="unsloth",
)

# Load domain dataset (example: medical Q&A)
dataset = load_dataset("medalpaca/medical_meadow_medqa", split="train")

domain_prompt = """You are a domain expert. Answer the following question accurately.

### Question:
{question}

### Context:
{context}

### Answer:
{answer}"""

def format_domain(examples):
    texts = []
    for q, c, a in zip(
        examples["question"],
        examples.get("context", [""] * len(examples["question"])),
        examples["answer"]
    ):
        texts.append(domain_prompt.format(
            question=q, context=c or "N/A", answer=a
        ) + tokenizer.eos_token)
    return {"text": texts}

dataset = dataset.map(format_domain, batched=True)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,          # More epochs for domain data
        learning_rate=1e-4,          # Lower LR for domain knowledge
        fp16=True,
        logging_steps=1,
        output_dir="outputs/domain-model",
    ),
)

trainer.train()
```

## Use Case 5: Vision Model Fine-Tuning

**Goal**: Fine-tune multimodal models that process both images and text.

### Training Script

```python
from unsloth import FastVisionModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Load vision model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Llama-3.2-11B-Vision-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastVisionModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    finetune_vision_layers=True,     # Also fine-tune vision encoder
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)

# Load image-text dataset
dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=1,
        output_dir="outputs/vision-model",
    ),
)

trainer.train()
```

## Use Case 6: Embedding Model Fine-Tuning

**Goal**: Fine-tune sentence transformer models for semantic search and retrieval.

### Training Script

```python
from unsloth import FastLanguageModel
from sentence_transformers import SentenceTransformerTrainer
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B",
    max_seq_length=512,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)

# Load pairs dataset for contrastive learning
dataset = load_dataset("sentence-transformers/all-nli", split="train")

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=16,
        num_train_epochs=1,
        learning_rate=2e-5,
        output_dir="outputs/embedding-model",
    ),
)

trainer.train()
```

## Advanced Configuration

### LoRA Hyperparameters

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `r` | LoRA rank | 8-128 (higher = more capacity) |
| `lora_alpha` | Scaling factor | Same as `r` or 2x `r` |
| `lora_dropout` | Regularization | 0 (SFT), 0.05-0.1 (small datasets) |
| `target_modules` | Layers to adapt | All projection layers recommended |

### Training Hyperparameters

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `learning_rate` | Step size | 1e-5 to 5e-4 |
| `batch_size` | Samples per step | 1-8 (depends on VRAM) |
| `gradient_accumulation` | Virtual batch size | 4-16 |
| `num_epochs` | Training passes | 1-5 |
| `warmup_steps` | LR warmup | 5-100 |
| `max_seq_length` | Token limit | 512-131072 |

### Memory Optimization

```python
# Enable gradient checkpointing (saves ~60% VRAM)
use_gradient_checkpointing="unsloth"

# Use 4-bit quantization
load_in_4bit=True

# Enable sequence packing (faster training)
packing=True
```

## Model Selection Guide

| Model | Parameters | VRAM (4-bit) | Best For |
|-------|-----------|-------------|----------|
| Llama 3.2 1B | 1B | ~2GB | Edge devices, quick experiments |
| Qwen 2.5 3B | 3B | ~4GB | General tasks, chatbots |
| Gemma 3 4B | 4B | ~5GB | Instruction following |
| Llama 3.1 8B | 8B | ~6GB | Production chatbots |
| Qwen 2.5 14B | 14B | ~10GB | Complex reasoning |
| Llama 3.1 70B | 70B | ~42GB | Enterprise, multi-GPU |

## Next Steps

- [RL Training Guide](RL_TRAINING_GUIDE.md) — GRPO and reward-based training
- [Data Preparation](DATA_PREPARATION.md) — Dataset formatting best practices
- [Model Export Guide](MODEL_EXPORT_GUIDE.md) — Deploy your fine-tuned model
