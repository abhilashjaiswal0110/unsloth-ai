# Local Development Guide

This guide covers setting up a development environment for contributing to Unsloth.

## Prerequisites

- **Python**: 3.9 – 3.14
- **Git**: 2.30+
- **GPU**: NVIDIA with CUDA 11.8+ (recommended for testing kernels)
- **Node.js**: 18+ (only for Studio frontend development)

## Repository Setup

### 1. Clone the Repository

```bash
git clone https://github.com/abhilashjaiswal0110/unsloth-ai.git
cd unsloth-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Install in Development Mode

```bash
# Core package (no GPU dependencies)
pip install -e .

# With HuggingFace/training support
pip install -e ".[huggingface]"

# With Triton kernels (Linux, requires GPU)
pip install -e ".[triton]"
```

### 4. Install Development Tools

```bash
pip install ruff pytest pytest-cov pre-commit
pre-commit install
```

## Project Structure

```
unsloth-ai/
├── unsloth/                # Main Python package
│   ├── models/             # Model implementations (Llama, Qwen, Gemma, etc.)
│   ├── kernels/            # Custom Triton/CUDA kernels
│   ├── dataprep/           # Dataset preprocessing utilities
│   ├── registry/           # Model auto-discovery and registration
│   ├── utils/              # Shared utilities
│   ├── trainer.py          # Training orchestration
│   ├── save.py             # Model serialization
│   ├── chat_templates.py   # Chat template definitions
│   └── __init__.py         # Package entry, import fixes
├── cli/                    # CLI commands (train, inference, export)
├── studio/                 # Web UI (backend + React frontend)
│   ├── backend/            # Python backend services
│   └── frontend/           # React/Vite frontend
├── tests/                  # Test suite
│   ├── utils/              # Test utilities & evaluations
│   ├── qlora/              # QLoRA-specific tests
│   └── saving/             # Model saving tests
├── agents/                 # AI agent skills and plugins
├── docs/                   # Documentation
├── scripts/                # Build & utility scripts
├── pyproject.toml          # Project configuration
└── build.sh                # Build automation script
```

## Development Workflow

### Running Linters

```bash
# Check code style
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .

# Run pre-commit on all files
pre-commit run --all-files
```

### Running Tests

```bash
# Run all unit tests (CPU-only)
pytest tests/ -v --tb=short -k "not cuda and not gpu"

# Run specific test file
pytest tests/test_model_registry.py -v

# Run with coverage
pytest tests/ --cov=unsloth --cov-report=html

# Run QLoRA tests (requires GPU)
pytest tests/qlora/ -v
```

### Building the Package

```bash
# Build wheel
python -m build

# Build including Studio frontend
bash build.sh
```

### Studio Development

```bash
# Backend (from repo root)
cd studio
python -m studio.backend

# Frontend (requires Node.js 18+)
cd studio/frontend
npm install
npm run dev       # Development server at localhost:5173
npm run build     # Production build
```

## Code Style Guidelines

- **Formatter**: Ruff (configured in `pyproject.toml` and `.pre-commit-config.yaml`)
- **Line Length**: Follow existing patterns in each file
- **Docstrings**: Follow existing conventions in the codebase
- **Type Hints**: Preferred but not strictly enforced
- **Imports**: Sorted by Ruff, grouped by standard/third-party/local

## Git Workflow

### Branch Naming

```
feature/description     # New features
fix/description         # Bug fixes
docs/description        # Documentation updates
refactor/description    # Code refactoring
```

### Commit Messages

Use conventional commit format:

```
feat: add support for new model architecture
fix: resolve VRAM leak during training
docs: update fine-tuning guide
chore: update dependencies
test: add registry validation tests
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes with tests
4. Run linters and tests locally
5. Submit a PR using the PR template
6. Address review feedback from code owners

## Environment Variables

Create a `.env` file for local configuration (already in `.gitignore`):

```bash
# HuggingFace Hub
HF_TOKEN=your_huggingface_token

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key

# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# Unsloth Settings
UNSLOTH_IS_PRESENT=1
```

## Debugging

### GPU Memory Issues

```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
torch.cuda.empty_cache()
```

### Model Loading Issues

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose model loading
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

## Common Development Tasks

### Adding a New Model

1. Create model file in `unsloth/models/` (follow `llama.py` as reference)
2. Register in `unsloth/registry/` with a new `_modelname.py` file
3. Add chat template mappings in `unsloth/chat_templates.py`
4. Add tests in `tests/`
5. Update documentation

### Adding a New Kernel

1. Create kernel file in `unsloth/kernels/`
2. Register in `unsloth/kernels/__init__.py`
3. Add benchmarks and tests
4. Document performance characteristics

### Adding a CLI Command

1. Create command in `cli/commands/`
2. Register in `cli/commands/__init__.py`
3. Add options in `cli/options.py`
4. Update `docs/API_REFERENCE.md`
