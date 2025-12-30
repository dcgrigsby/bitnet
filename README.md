# BitNet b1.58 Training Implementation

A PyTorch implementation of **BitNet b1.58** - a transformer-based architecture with ternary weight quantization (1.58 bits) optimized for CPU inference. By quantizing weights to {-1, 0, 1}, BitNet replaces expensive matrix multiplications with simple addition operations, reducing computation.

This is an **implementation of the training pipeline** for BitNet, based on the [BitNet: Scaling Bitwise Operations for Efficient Transformer Inference and Learning](https://arxiv.org/abs/2310.11453) paper (Ma et al., 2023).

**Current status**: Training infrastructure is complete and working. Inference capabilities are available for model evaluation and testing (e.g., chat, loss monitoring), but the optimized BitNet kernels for efficient inference are not yet implemented.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. 

Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate  # or `direnv allow` if using direnv
uv sync
```

## Commands

This project uses [just](https://github.com/casey/just) for command definitions. 

Run `just` to see available commands, or check the `justfile` for details.

## Quick Start

### Run Full Experiments

Each `exp-*` command runs a complete experiment with sensible defaults, provides setup information, and produces trained checkpoints:

```bash
just exp-arithmetic     # 5M params, arithmetic task (~30 min)
just exp-tinystories    # 12M params, natural language (~2 hours)
just exp-baseline       # 95M params, FineWeb-Edu dataset (~3-4 days)
```

For custom training arguments, run the Python scripts directly:

```bash
uv run python experiments/arithmetic/train_bitnet_arithmetic.py --num-steps 5000 --batch-size 64
```

See experiment-specific configs in `experiments/*/TRAINING_*_CONFIG.md` for all available options.

## Using Trained Models

### Chat with a Model

After training, chat interactively with the latest trained model:

```bash
just chat
```

Or specify a checkpoint explicitly:

```bash
just chat --checkpoint runs/bitnet_95M_<timestamp>/checkpoints/step_010000/checkpoint.pt
```

### Monitor Training Progress

```bash
# Plot training loss curve
just plot-loss bitnet_5M_arithmetic

# Check training status
just check-status bitnet_5M_arithmetic
```

## Experiments

Each experiment demonstrates different aspects of BitNet training:

### Arithmetic (5M parameters)

- **Dataset**: Synthetic arithmetic problems (e.g., `2+3=5`)
- **Typical runtime on RTX-3060**: ~30 minutes on GPU
- **Expected loss**: Drops to ~0.5
- **Use case**: Quick validation of quantization; verify your setup works before larger experiments

### TinyStories (12M parameters)

- **Dataset**: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) - simple children's stories
- **Typical runtime on RTX-3060**: ~2 hours 
- **Expected loss**: Drops to ~1.5-2.0
- **Use case**: Natural language baseline; chat with trained models

### Baseline (95M parameters)

- **Dataset**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) subset
- **Typical runtime on RTX-3060**: ~77 hours 
- **Expected loss**: Drops to ~2.0-2.5
- **Use case**: Largest model that fits on RTX 3060
- **Note**: For practical use, target at least 1B parameters; 3B-7B optimal

## Project Structure

```
bitnet/
├── src/bitnet/           # Core BitNet implementation
│   ├── model.py          # BitNet architecture
│   ├── quantization.py   # Ternary quantization
│   └── ...
├── experiments/          # Self-contained experiment directories
│   ├── arithmetic/       # Arithmetic validation experiment
│   ├── tinystories/      # Natural language experiment
│   └── baseline-95m/     # Larger-scale baseline
├── scripts/                     # Utility scripts
│   ├── plot_loss.py             # Visualize training curves
│   ├── check_training_status.py # Monitor progress
│   └── chat_bitnet.py           # Interactive chat with models
├── docs/                 # Bitnet papers
├── runs/                 # Training outputs (created during training)
│   └── bitnet_<desc>_<timestamp>/
│       ├── checkpoints/
│       ├── config.yaml
│       └── logs/
├── tests/                # Test suite
├── justfile              # Command definitions
├── pyproject.toml        # Project metadata and dependencies
├── .envrc                # Direnv configuration
└── README.md             # This file
```
