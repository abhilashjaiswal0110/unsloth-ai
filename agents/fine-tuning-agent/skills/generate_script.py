"""
Fine-Tuning Agent: Generate Script Skill

Generates complete, ready-to-run training scripts from configuration.
"""


def generate_training_script(config: dict) -> str:
    """
    Generate a complete fine-tuning script from configuration.

    Args:
        config: Configuration dictionary from recommend_config

    Returns:
        Complete Python training script as a string
    """
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    dataset_cfg = config["dataset"]

    target_modules_str = ", ".join(f'"{m}"' for m in lora_cfg["target_modules"])

    script = f'''"""
Auto-generated Unsloth Fine-Tuning Script
Model: {model_cfg["name"]}
Task: {dataset_cfg["task_type"]}
"""

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ============================================================
# Step 1: Load Model
# ============================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{model_cfg["name"]}",
    max_seq_length={model_cfg["max_seq_length"]},
    load_in_4bit={model_cfg["load_in_4bit"]},
)

# ============================================================
# Step 2: Configure LoRA
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r={lora_cfg["r"]},
    target_modules=[{target_modules_str}],
    lora_alpha={lora_cfg["lora_alpha"]},
    lora_dropout={lora_cfg["lora_dropout"]},
    use_gradient_checkpointing="{lora_cfg["use_gradient_checkpointing"]}",
)

# ============================================================
# Step 3: Load & Format Dataset
# ============================================================
dataset = load_dataset("{dataset_cfg["name"]}", split="train")

prompt_template = """### Instruction:
{{instruction}}

### Input:
{{input}}

### Response:
{{output}}"""

def format_prompts(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        text = prompt_template.format(
            instruction=examples["instruction"][i],
            input=examples.get("input", [""] * len(examples["instruction"]))[i],
            output=examples["output"][i],
        )
        texts.append(text + tokenizer.eos_token)
    return {{"text": texts}}

dataset = dataset.map(format_prompts, batched=True)

# ============================================================
# Step 4: Train
# ============================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length={model_cfg["max_seq_length"]},
    packing={train_cfg["packing"]},
    args=TrainingArguments(
        per_device_train_batch_size={train_cfg["per_device_train_batch_size"]},
        gradient_accumulation_steps={train_cfg["gradient_accumulation_steps"]},
        warmup_steps={train_cfg["warmup_steps"]},
        num_train_epochs={train_cfg["num_train_epochs"]},
        learning_rate={train_cfg["learning_rate"]},
        fp16={train_cfg["fp16"]},
        logging_steps={train_cfg["logging_steps"]},
        output_dir="outputs",
        seed={train_cfg["seed"]},
    ),
)

print("Starting training...")
trainer.train()
print("Training complete!")

# ============================================================
# Step 5: Save Model
# ============================================================
model.save_pretrained("finetuned-model")
tokenizer.save_pretrained("finetuned-model")
print("Model saved to ./finetuned-model")
'''

    return script


def validate_config(config: dict) -> list[str]:
    """
    Validate a training configuration and return warnings.

    Args:
        config: Configuration dictionary

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    hardware = config.get("hardware", {})

    # Check VRAM constraints
    estimated = hardware.get("estimated_vram_usage_gb", 0)
    available = hardware.get("gpu_vram_gb", 0)
    if estimated > available * 0.9:
        warnings.append(
            f"Estimated VRAM usage ({estimated:.1f} GB) is close to "
            f"available VRAM ({available} GB). Consider reducing batch size or LoRA rank."
        )

    # Check learning rate
    lr = train_cfg.get("learning_rate", 0)
    if lr > 5e-4:
        warnings.append(
            f"Learning rate ({lr}) is high. This may cause unstable training."
        )

    # Check LoRA rank
    r = lora_cfg.get("r", 0)
    if r > 128:
        warnings.append(
            f"LoRA rank ({r}) is very high. Consider r=64 or lower for efficiency."
        )

    # Check sequence length
    seq_len = model_cfg.get("max_seq_length", 0)
    if seq_len > 8192:
        warnings.append(
            f"Sequence length ({seq_len}) is very long. This significantly increases VRAM usage."
        )

    return warnings
