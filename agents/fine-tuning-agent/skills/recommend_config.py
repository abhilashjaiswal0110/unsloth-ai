"""
Fine-Tuning Agent: Recommend Config Skill

Recommends optimal model and training configuration based on
hardware constraints, dataset characteristics, and task requirements.
"""

# Model VRAM requirements (approximate, 4-bit quantization, training)
MODEL_VRAM_MAP = {
    "1B": 4,
    "3B": 6,
    "4B": 7,
    "7B": 10,
    "8B": 12,
    "13B": 18,
    "14B": 20,
    "70B": 80,
}

# Recommended models by task type
TASK_MODEL_MAP = {
    "instruction": [
        "unsloth/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.2-1B-Instruct",
        "unsloth/Qwen2.5-7B-Instruct",
        "unsloth/Gemma-3-4B-it",
    ],
    "chat": [
        "unsloth/Qwen2.5-7B-Instruct",
        "unsloth/Llama-3.1-8B-Instruct",
        "unsloth/Gemma-3-4B-it",
    ],
    "code": [
        "unsloth/Qwen2.5-Coder-7B-Instruct",
        "unsloth/Qwen2.5-Coder-3B-Instruct",
    ],
    "reasoning": [
        "unsloth/Qwen2.5-14B-Instruct",
        "unsloth/Qwen2.5-7B-Instruct",
        "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    ],
    "domain": [
        "unsloth/Gemma-3-4B-it",
        "unsloth/Llama-3.2-3B-Instruct",
        "unsloth/Qwen2.5-7B-Instruct",
    ],
}


def _estimate_model_size(model_name: str) -> str:
    """Extract model size from model name."""
    import re

    match = re.search(r"(\d+)[Bb]", model_name)
    if match:
        return f"{match.group(1)}B"
    return "7B"  # Default assumption


def _get_vram_requirement(model_size: str) -> int:
    """Get estimated VRAM requirement for training."""
    return MODEL_VRAM_MAP.get(model_size, 12)


def recommend_config(
    model_name: str = None,
    dataset_name: str = None,
    gpu_vram_gb: int = 24,
    task_type: str = "instruction",
    max_seq_length: int = 2048,
    dataset_size: int = None,
) -> dict:
    """
    Recommend optimal fine-tuning configuration.

    Args:
        model_name: Specific model to use (auto-selected if None)
        dataset_name: HuggingFace dataset name or local path
        gpu_vram_gb: Available GPU VRAM in GB
        task_type: One of 'instruction', 'chat', 'code', 'reasoning', 'domain'
        max_seq_length: Maximum sequence length
        dataset_size: Approximate dataset size (number of samples)

    Returns:
        Configuration dictionary with recommended parameters
    """
    # Auto-select model if not specified
    if model_name is None:
        candidates = TASK_MODEL_MAP.get(task_type, TASK_MODEL_MAP["instruction"])
        for candidate in candidates:
            size = _estimate_model_size(candidate)
            if _get_vram_requirement(size) <= gpu_vram_gb:
                model_name = candidate
                break
        if model_name is None:
            model_name = candidates[-1]  # Smallest model as fallback

    model_size = _estimate_model_size(model_name)
    vram_needed = _get_vram_requirement(model_size)

    # Determine LoRA rank based on available VRAM headroom
    vram_headroom = gpu_vram_gb - vram_needed
    if vram_headroom >= 8:
        lora_r = 64
    elif vram_headroom >= 4:
        lora_r = 32
    elif vram_headroom >= 2:
        lora_r = 16
    else:
        lora_r = 8

    # Determine batch size
    if vram_headroom >= 6:
        batch_size = 4
    elif vram_headroom >= 3:
        batch_size = 2
    else:
        batch_size = 1

    # Determine epochs based on dataset size
    if dataset_size is not None:
        if dataset_size < 1000:
            num_epochs = 5
        elif dataset_size < 10000:
            num_epochs = 3
        elif dataset_size < 100000:
            num_epochs = 1
        else:
            num_epochs = 1
    else:
        num_epochs = 1

    # Learning rate based on task type
    lr_map = {
        "instruction": 2e-4,
        "chat": 2e-4,
        "code": 1e-4,
        "reasoning": 5e-5,
        "domain": 1e-4,
    }
    learning_rate = lr_map.get(task_type, 2e-4)

    config = {
        "model": {
            "name": model_name,
            "max_seq_length": max_seq_length,
            "load_in_4bit": True,
        },
        "lora": {
            "r": lora_r,
            "lora_alpha": lora_r,
            "lora_dropout": 0,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "use_gradient_checkpointing": "unsloth",
        },
        "training": {
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": max(1, 8 // batch_size),
            "learning_rate": learning_rate,
            "num_train_epochs": num_epochs,
            "warmup_steps": 10,
            "fp16": True,
            "logging_steps": 1,
            "packing": True,
            "seed": 42,
        },
        "dataset": {
            "name": dataset_name,
            "task_type": task_type,
        },
        "hardware": {
            "gpu_vram_gb": gpu_vram_gb,
            "estimated_vram_usage_gb": vram_needed + (lora_r * 0.1),
        },
    }

    return config
