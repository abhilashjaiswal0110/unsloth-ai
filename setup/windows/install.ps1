#Requires -Version 5.1
<#
.SYNOPSIS
    Isolated Windows environment setup for Unsloth Qwen3 Advanced GRPO training.

.DESCRIPTION
    Creates a dedicated conda environment (unsloth-qwen3-grpo) that does NOT
    interfere with any existing Python or PyTorch installations on the system.

    Steps performed:
      1.  Verify NVIDIA GPU + CUDA driver
      2.  Install Miniconda if conda is absent
      3.  Create / recreate the isolated conda environment
      4.  Install PyTorch (CUDA 12.4 by default) + all ML dependencies
      5.  Install Unsloth from source in editable mode
      6.  Install triton-windows, bitsandbytes, xformers (Windows-specific)
      7.  Write a .env template for credentials / tuning knobs
      8.  Run a quick smoke-test to verify the stack

.NOTES
    Prerequisites:
      - NVIDIA GPU (CUDA 11.8+ driver; CUDA 12.4 driver recommended)
      - Internet access
      - Run from the repository root:
          powershell -ExecutionPolicy Bypass -File setup\windows\install.ps1

    Supported Python: 3.11 (default)
    Supported CUDA:   11.8 | 12.1 | 12.4 (default)
    Conda env name:   unsloth-qwen3-grpo
#>

[CmdletBinding()]
param(
    [string]$CudaVersion   = "12.4",           # "11.8" | "12.1" | "12.4"
    [string]$PythonVersion = "3.11",            # "3.10" | "3.11" | "3.12"
    [string]$EnvName       = "unsloth-qwen3-grpo",
    [switch]$Reinstall                          # Force recreate environment
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"   # Faster Invoke-WebRequest

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
function Write-Header  { param($msg) Write-Host "`n══ $msg ══" -ForegroundColor Cyan   }
function Write-Success { param($msg) Write-Host "✔  $msg"     -ForegroundColor Green  }
function Write-Warn    { param($msg) Write-Host "⚠  $msg"     -ForegroundColor Yellow }
function Write-Fail    { param($msg) Write-Host "✘  $msg"     -ForegroundColor Red    }
function Write-Step    { param($msg) Write-Host "→  $msg"     -ForegroundColor White  }

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Locate repo root (script lives at <root>\setup\windows\install.ps1)
# ─────────────────────────────────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

Write-Header "Unsloth Qwen3 Advanced GRPO — Windows Isolated Setup"
Write-Step  "Repository root : $RepoRoot"
Write-Step  "CUDA version    : $CudaVersion"
Write-Step  "Python version  : $PythonVersion"
Write-Step  "Conda env name  : $EnvName"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  NVIDIA GPU check
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "Step 1 — Verifying NVIDIA GPU"

try {
    $nvsmiPath = (Get-Command nvidia-smi -ErrorAction SilentlyContinue)?.Source
    if (-not $nvsmiPath) {
        $nvsmiPath = "C:\Windows\System32\nvidia-smi.exe"
    }
    $gpuInfo = & $nvsmiPath --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi returned non-zero exit" }
    Write-Success "GPU detected:`n$gpuInfo"
} catch {
    Write-Fail "NVIDIA GPU not found or driver not installed."
    Write-Warn "Install NVIDIA drivers from: https://www.nvidia.com/drivers"
    exit 1
}

# Parse driver version to warn on very old drivers
$driverLine = ($gpuInfo -split "`n")[0]
$driverVer  = ($driverLine -split ",")[1].Trim()
Write-Step "Driver version: $driverVer"

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Conda availability — install Miniconda if missing
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "Step 2 — Checking for Conda"

function Find-Conda {
    # Try conda on PATH first
    $c = Get-Command conda -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }

    # Common install locations
    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\Anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

$condaExe = Find-Conda

if (-not $condaExe) {
    Write-Warn "Conda not found. Installing Miniconda3 (user-only, isolated)..."

    $minicondaUrl      = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    $minicondaInstaller = "$env:TEMP\Miniconda3-latest-Windows-x86_64.exe"
    $minicondaTarget    = "$env:USERPROFILE\miniconda3"

    Write-Step "Downloading Miniconda..."
    Invoke-WebRequest -Uri $minicondaUrl -OutFile $minicondaInstaller -UseBasicParsing

    Write-Step "Installing Miniconda to $minicondaTarget ..."
    Start-Process -FilePath $minicondaInstaller -ArgumentList @(
        "/S", "/D=$minicondaTarget"
    ) -Wait -NoNewWindow

    $condaExe = "$minicondaTarget\Scripts\conda.exe"
    if (-not (Test-Path $condaExe)) {
        Write-Fail "Miniconda installation failed. Install manually from:"
        Write-Fail "  https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    }

    # Reload PATH so conda is found in subsequent calls
    $env:PATH = "$minicondaTarget\Scripts;$minicondaTarget;$env:PATH"
    Write-Success "Miniconda installed at $minicondaTarget"
} else {
    Write-Success "Conda found: $condaExe"
}

# Ensure conda is on PATH for the rest of this session
$condaDir = Split-Path -Parent $condaExe
if ($env:PATH -notlike "*$condaDir*") {
    $env:PATH = "$condaDir;$env:PATH"
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Create (or recreate) the isolated environment
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "Step 3 — Creating Isolated Conda Environment: $EnvName"

# Check if env already exists
$existingEnvs = & $condaExe env list --json 2>&1 | ConvertFrom-Json
$envExists    = $existingEnvs.envs | Where-Object { $_ -match [regex]::Escape($EnvName) }

if ($envExists -and $Reinstall) {
    Write-Warn "Removing existing environment '$EnvName' (--Reinstall flag)..."
    & $condaExe env remove --name $EnvName --yes 2>&1 | Out-Null
    $envExists = $null
}

if ($envExists) {
    Write-Success "Environment '$EnvName' already exists. Skipping creation."
    Write-Warn  "Run with -Reinstall to recreate from scratch."
} else {
    Write-Step "Creating environment from: setup\windows\environment.yml"
    $envYml = Join-Path $RepoRoot "setup\windows\environment.yml"

    # Override python version dynamically
    $yamlContent = Get-Content $envYml -Raw
    $yamlContent = $yamlContent -replace "python=\d+\.\d+", "python=$PythonVersion"
    $tmpYml      = "$env:TEMP\unsloth_env_tmp.yml"
    $yamlContent | Set-Content $tmpYml -Encoding UTF8

    & $condaExe env create --file $tmpYml --name $EnvName 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Environment creation failed. Check the output above."
        exit 1
    }
    Write-Success "Environment '$EnvName' created."
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run a command inside the conda environment
# ─────────────────────────────────────────────────────────────────────────────
function Invoke-InEnv {
    param([string]$Command, [string[]]$Args, [switch]$NoThrow)

    $condaRun = @("run", "--name", $EnvName, "--no-capture-output", $Command) + $Args
    Write-Step "conda $($condaRun -join ' ')"
    & $condaExe @condaRun
    if ($LASTEXITCODE -ne 0 -and -not $NoThrow) {
        Write-Fail "Command failed: $Command $($Args -join ' ')"
        exit 1
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Install PyTorch with correct CUDA version
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "Step 4 — Installing PyTorch (CUDA $CudaVersion)"

$cudaTag = "cu" + ($CudaVersion -replace "\.", "")   # e.g. "cu124"
$torchIndex = "https://download.pytorch.org/whl/$cudaTag"

Invoke-InEnv pip @(
    "install", "--upgrade",
    "torch", "torchvision", "torchaudio",
    "--index-url", $torchIndex
)
Write-Success "PyTorch installed with CUDA $CudaVersion support."

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Install Windows-specific packages (triton-windows, bitsandbytes, xformers)
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "Step 5 — Installing Windows-Specific ML Packages"

Write-Step "Installing triton-windows..."
Invoke-InEnv pip @("install", "triton-windows")

Write-Step "Installing bitsandbytes (Windows)..."
Invoke-InEnv pip @("install", "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0")

Write-Step "Installing xformers..."
Invoke-InEnv pip @("install", "xformers>=0.0.28", "--index-url", $torchIndex)

Write-Success "Windows-specific ML packages installed."

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Install HuggingFace ecosystem & training libs
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "Step 6 — Installing HuggingFace Ecosystem"

Invoke-InEnv pip @(
    "install", "--upgrade",
    "transformers>=4.51.3,!=4.52.0,!=4.52.1,!=4.52.2,!=4.52.3",
    "accelerate>=0.34.1",
    "peft>=0.18.0,!=0.11.0",
    "trl>=0.18.2,!=0.19.0,<=0.24.0",
    "datasets>=3.4.1,!=4.0.*,!=4.1.0,<4.4.0",
    "huggingface_hub>=0.34.0",
    "hf_transfer",
    "diffusers",
    "sentence-transformers",
    "tokenizers",
    "safetensors",
    "sentencepiece>=0.2.0"
)

Write-Step "Installing unsloth_zoo..."
Invoke-InEnv pip @("install", "unsloth_zoo>=2026.3.2")

Write-Success "HuggingFace ecosystem installed."

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Install Unsloth from source (editable mode)
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "Step 7 — Installing Unsloth from Source (Editable)"

# Install core unsloth without overriding torch
Invoke-InEnv pip @(
    "install", "-e", "$RepoRoot",
    "--no-deps"                    # deps already installed; avoid overrides
)

Write-Success "Unsloth installed from source in editable mode."

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Install auxiliary & logging packages
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "Step 8 — Installing Auxiliary Packages"

Invoke-InEnv pip @(
    "install",
    "wandb", "tensorboard",
    "sympy", "antlr4-python3-runtime==4.11",
    "tyro", "pyyaml", "pydantic", "nest-asyncio",
    "numpy", "tqdm", "psutil", "packaging", "protobuf",
    "ruff", "pytest", "ipykernel", "ipywidgets"
)

Write-Success "Auxiliary packages installed."

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Write .env template for credentials & config
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "Step 9 — Writing .env Template"

$envFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envFile)) {
    @"
# ─── HuggingFace ─────────────────────────────────────────────────────────────
HF_TOKEN=your_huggingface_token_here
HF_HUB_ENABLE_HF_TRANSFER=1          # Faster model downloads via hf_transfer

# ─── Weights & Biases (optional) ─────────────────────────────────────────────
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=qwen3-grpo
WANDB_ENTITY=your_team_or_username

# ─── GPU & CUDA ───────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0               # Set to "0,1" for multi-GPU

# ─── Unsloth ──────────────────────────────────────────────────────────────────
UNSLOTH_IS_PRESENT=1
UNSLOTH_OFFLOAD_TO_DISK=0            # Set 1 to offload optimizer states

# ─── Training defaults (override via CLI or YAML) ─────────────────────────────
GRPO_MODEL=unsloth/Qwen3-8B
GRPO_OUTPUT_DIR=outputs/qwen3-grpo
"@ | Set-Content $envFile -Encoding UTF8
    Write-Success ".env template written to $envFile"
} else {
    Write-Warn ".env already exists — skipping template write."
}

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Smoke-test the installation
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "Step 10 — Smoke-Test"

$testScript = @"
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import transformers, peft, trl, accelerate, datasets
print(f'transformers: {transformers.__version__}')
print(f'peft:         {peft.__version__}')
print(f'trl:          {trl.__version__}')
print(f'accelerate:   {accelerate.__version__}')
print(f'datasets:     {datasets.__version__}')

import unsloth
print(f'unsloth:      {unsloth.__version__}')

import bitsandbytes
print(f'bitsandbytes: {bitsandbytes.__version__}')

try:
    import triton
    print(f'triton:       {triton.__version__}')
except ImportError:
    print('triton:       not available (optional on Windows)')

print()
print('✔  All core packages imported successfully.')
"@

$testFile = "$env:TEMP\unsloth_smoke_test.py"
$testScript | Set-Content $testFile -Encoding UTF8

Invoke-InEnv python @($testFile) -NoThrow

if ($LASTEXITCODE -ne 0) {
    Write-Fail "Smoke-test failed — review errors above."
    exit 1
}

Write-Success "Smoke-test passed."

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
Write-Header "Setup Complete"

Write-Host @"

  ┌──────────────────────────────────────────────────────────────────┐
  │  Environment : $EnvName
  │  Activate    : conda activate $EnvName
  │  Train       : python scripts\train_qwen3_grpo.py --config configs\qwen3_grpo.yaml
  │  Docs        : docs\WINDOWS_QWEN3_GRPO_SETUP.md
  └──────────────────────────────────────────────────────────────────┘

  Edit .env with your HF_TOKEN and optional WANDB_API_KEY before training.
"@ -ForegroundColor Cyan
