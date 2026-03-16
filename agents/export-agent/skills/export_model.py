"""
Export Agent: Export Model Skill

Exports fine-tuned models to various deployment formats
with optimal quantization settings.
"""

from pathlib import Path


# Quantization quality/size trade-offs
QUANTIZATION_INFO = {
    "q2_k": {"quality": "low", "size_ratio": 0.18, "use_case": "Edge/mobile, minimum size"},
    "q4_0": {"quality": "good", "size_ratio": 0.27, "use_case": "Fast inference, small footprint"},
    "q4_k_m": {"quality": "good+", "size_ratio": 0.29, "use_case": "Recommended default"},
    "q5_k_m": {"quality": "high", "size_ratio": 0.36, "use_case": "Quality-focused deployment"},
    "q8_0": {"quality": "very high", "size_ratio": 0.50, "use_case": "Maximum quality"},
    "f16": {"quality": "lossless", "size_ratio": 1.00, "use_case": "Reference/benchmarking"},
}

# Chat templates for common model families
OLLAMA_TEMPLATES = {
    "llama": '''TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|eot_id|>"''',
    "qwen": '''TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""

PARAMETER stop "<|im_end|>"''',
    "gemma": '''TEMPLATE """{{ if .System }}<start_of_turn>user
{{ .System }}

{{ .Prompt }}<end_of_turn>
{{ else }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>"""

PARAMETER stop "<end_of_turn>"''',
}


def recommend_quantization(
    model_size_b: float,
    target_vram_gb: float = None,
    priority: str = "balanced",
) -> str:
    """
    Recommend optimal quantization method.

    Args:
        model_size_b: Model size in billions of parameters
        target_vram_gb: Target VRAM budget in GB
        priority: 'quality', 'balanced', or 'size'

    Returns:
        Recommended quantization method string
    """
    if priority == "quality":
        return "q8_0"
    elif priority == "size":
        return "q4_0"

    # Balanced: check if we have a VRAM target
    if target_vram_gb is not None:
        model_f16_gb = model_size_b * 2  # ~2GB per billion params at f16
        for quant, info in QUANTIZATION_INFO.items():
            estimated_size = model_f16_gb * info["size_ratio"] * 1.2  # 20% overhead
            if estimated_size <= target_vram_gb:
                return quant
        return "q2_k"  # Smallest if nothing fits

    return "q4_k_m"  # Default recommendation


def generate_modelfile(
    gguf_path: str,
    model_family: str = "llama",
    system_prompt: str = None,
    temperature: float = 0.7,
) -> str:
    """
    Generate an Ollama Modelfile for the exported model.

    Args:
        gguf_path: Path to the GGUF file
        model_family: Model family (llama, qwen, gemma)
        system_prompt: Optional system prompt
        temperature: Default temperature

    Returns:
        Modelfile content as a string
    """
    template = OLLAMA_TEMPLATES.get(model_family, OLLAMA_TEMPLATES["llama"])

    modelfile = f"FROM {gguf_path}\n\n"
    modelfile += template + "\n\n"
    modelfile += f"PARAMETER temperature {temperature}\n"
    modelfile += "PARAMETER top_p 0.9\n"

    if system_prompt:
        modelfile += f'\nSYSTEM """{system_prompt}"""\n'

    return modelfile


def generate_export_script(
    model_path: str,
    output_dir: str,
    export_format: str = "gguf",
    quantization: str = "q4_k_m",
    model_family: str = "llama",
    hub_repo: str = None,
) -> str:
    """
    Generate a complete export script.

    Args:
        model_path: Path to the fine-tuned model
        output_dir: Output directory for exported files
        export_format: Export format (gguf, merged, hub)
        quantization: GGUF quantization method
        model_family: Model family for chat template
        hub_repo: HuggingFace Hub repo ID (for hub export)

    Returns:
        Complete Python export script as a string
    """
    output_path = Path(output_dir)

    if export_format == "gguf":
        script = f'''"""
Auto-generated Export Script - GGUF Format
Quantization: {quantization}
"""

from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{model_path}",
    max_seq_length=2048,
)

# Export to GGUF
print("Exporting to GGUF ({quantization})...")
model.save_pretrained_gguf(
    "{output_path}",
    tokenizer,
    quantization_method="{quantization}",
)
print(f"GGUF model saved to {output_path}/")

# Generate Ollama Modelfile
modelfile_content = """FROM {output_path}/unsloth.{quantization.upper()}.gguf

{OLLAMA_TEMPLATES.get(model_family, OLLAMA_TEMPLATES["llama"])}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
"""

with open("{output_path}/Modelfile", "w") as f:
    f.write(modelfile_content)

print(f"Ollama Modelfile saved to {output_path}/Modelfile")
print(f"\\nTo use with Ollama:")
print(f"  ollama create my-model -f {output_path}/Modelfile")
print(f"  ollama run my-model")
'''

    elif export_format == "merged":
        script = f'''"""
Auto-generated Export Script - Merged 16-bit
"""

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{model_path}",
    max_seq_length=2048,
)

print("Merging and saving 16-bit model...")
model.save_pretrained_merged(
    "{output_path}",
    tokenizer,
    save_method="merged_16bit",
)
print(f"Merged model saved to {output_path}/")
'''

    elif export_format == "hub":
        if not hub_repo:
            hub_repo = "your-username/model-name"
        script = f'''"""
Auto-generated Export Script - HuggingFace Hub
"""

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{model_path}",
    max_seq_length=2048,
)

# Push LoRA adapters
print("Pushing LoRA adapters to Hub...")
model.push_to_hub("{hub_repo}")
tokenizer.push_to_hub("{hub_repo}")
print(f"Model pushed to https://huggingface.co/{hub_repo}")

# Optionally push GGUF versions
print("Pushing GGUF versions...")
model.push_to_hub_gguf(
    "{hub_repo}-GGUF",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0"],
)
print("GGUF versions pushed!")
'''
    else:
        script = f'# Unsupported format: {export_format}'

    return script
