# BitNet b1.58 Training Implementation

A PyTorch implementation of **BitNet b1.58** - a transformer-based architecture with ternary weight quantization (1.58 bits) for efficient neural network training.

This is an **implementation of the training pipeline** for BitNet, based on the [BitNet: Scaling Bitwise Operations for Efficient Transformer Inference and Learning](https://arxiv.org/abs/2310.11453) paper (Ma et al., 2023).

**Current status**: Training infrastructure is complete and working. Inference capabilities are available for model evaluation and testing (e.g., chat, tic-tac-toe gameplay), but the optimized BitNet kernels for efficient inference are not yet implemented.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and [just](https://github.com/casey/just) as a command runner. We recommend installing both, and optionally [direnv](https://direnv.net/) for automatic environment activation.

Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # or `direnv allow` if using direnv
uv sync
```

## Commands

Run `just` to see all available commands:
```bash
just
```

Commands are defined in the `justfile` and executed via `uv run`, so you don't need to manually manage the virtualenv.

## Quick Start

### Run a Quick Test
```bash
just train-simple 100
```
Trains a tiny model for 100 steps to verify everything is working (~1 minute).

### Run Full Experiments

Each experiment comes with sensible defaults and produces trained checkpoints:

```bash
just exp-arithmetic     # 5M params, arithmetic task (~30 min)
just exp-tictactoe      # 12M params, game learning (~60 min)
just exp-tinystories    # 12M params, natural language (~2 hours)
just exp-baseline       # 95M params, FineWeb-Edu dataset (~3-4 days)
```

### Run Training with Custom Arguments

```bash
# Arithmetic with custom steps and batch size
just train-arithmetic --num-steps 5000 --batch-size 64

# Tic-tac-toe with custom config
just train-tictactoe --num-steps 10000 --batch-size 64

# TinyStories with custom settings
just train-tinystories --num-steps 15000 --seq-len 128 --batch-size 16

# Baseline with custom run ID
just train-baseline --run-id my_experiment --batch-size 32
```

See experiment-specific configs in `experiments/*/TRAINING_*_CONFIG.md` for all available options.

## Using Trained Models

### Chat with a Model

After training, chat interactively with your trained model:

```bash
just chat runs/bitnet_95M_400k_<timestamp>/checkpoint_10000.pt
```

### Play Tic-Tac-Toe

If you trained the tic-tac-toe experiment, play against your trained model:

```bash
just play-tictactoe runs/bitnet_12M_tictactoe_<timestamp>/checkpoints/step_015000/checkpoint.pt
```

### Monitor Training Progress

```bash
# Plot training loss curve
just plot-loss bitnet_5M_arithmetic_1766885392

# Check training status (once)
just check-status bitnet_5M_arithmetic_1766885392

# Watch training status (updates every 10s)
just check-status bitnet_5M_arithmetic_1766885392 --watch
```

## Experiments

Each experiment demonstrates different aspects of BitNet training:

### Arithmetic (5M parameters)

- **Purpose**: Quick validation that BitNet quantization works correctly
- **Dataset**: Synthetic arithmetic problems (e.g., `2+3=5`)
- **Typical runtime**: ~30 minutes on modern GPU
- **Expected loss**: Drops from ~3.5 to ~0.5
- **Use case**: Verify your setup is working before trying larger experiments

### Tic-Tac-Toe (12M parameters)

- **Purpose**: Learn discrete game rules and strategy from synthetic data
- **Dataset**: Random-play tic-tac-toe games
- **Typical runtime**: ~60 minutes for 15k steps
- **Expected loss**: Drops from ~3-4 to ~0.5-1.0
- **Interactive**: Play against trained models!
- **Use case**: Proof that BitNet learns structured rules, play around with trained models

### TinyStories (12M parameters)

- **Purpose**: Validate natural language learning with real data
- **Dataset**: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) - simple children's stories
- **Typical runtime**: ~2 hours total (10 min tokenizer + 1.5 hrs training)
- **Expected loss**: Drops from ~7 to ~1.5-2.0
- **Use case**: Natural language baseline, chat with trained models

### Baseline (95M parameters)

- **Purpose**: Full-scale training baseline for research and comparison
- **Dataset**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) subset
- **Typical runtime**: ~77 hours (400k steps) on single A100
- **Config**: 16 batch size, 256 seq len, 2 gradient accumulation
- **Use case**: Research-grade baseline, infrastructure stress testing

## Project Structure

```
bitnet/
├── src/bitnet/              # Core BitNet implementation
│   ├── model.py            # BitNet architecture
│   ├── quantization.py     # Ternary quantization
│   └── ...
├── experiments/            # Self-contained experiment directories
│   ├── arithmetic/         # Arithmetic validation experiment
│   ├── tictactoe/         # Game learning experiment
│   ├── tinystories/       # Natural language experiment
│   └── baseline-95m/      # Large-scale baseline
├── scripts/               # Utility scripts
│   ├── plot_loss.py      # Visualize training curves
│   ├── check_training_status.py  # Monitor progress
│   └── chat_bitnet.py    # Interactive chat with models
├── docs/                 # Additional documentation
├── runs/                 # Training outputs (created during training)
│   └── bitnet_<desc>_<timestamp>/
│       ├── checkpoints/
│       ├── config.yaml
│       └── logs/
├── tests/               # Test suite
├── justfile            # Command definitions
├── pyproject.toml      # Project metadata and dependencies
├── .envrc              # Direnv configuration
└── README.md           # This file
```

## Development

### Run Tests

```bash
# Run all tests
just test

# Run specific test file
just test tests/test_model.py

# Run tests matching a pattern
just test -k "quantization"
```

## Technology Stack

- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[Transformers](https://huggingface.co/transformers/)**: Transformer utilities and pretrained models
- **[Datasets](https://huggingface.co/datasets/)**: HuggingFace datasets integration
- **[uv](https://docs.astral.sh/uv/)**: Python package management
- **[just](https://github.com/casey/just)**: Command runner
- **[direnv](https://direnv.net/)**: Automatic environment activation (optional)

## Training Tips

- **Start small**: Run `just train-simple 100` first to verify your setup
- **Monitor GPU**: Watch `nvidia-smi` to verify training is using your GPU
- **Check checkpoints**: Training saves checkpoints in `runs/` - you can resume from these or use them for inference
- **Customize hyperparameters**: Each experiment's training script accepts command-line arguments for batch size, learning rate, etc.
- **FineWeb-Edu dataset**: The baseline experiment will download ~20GB of data on first run

## Documentation

For more detailed information:

- `docs/BITNET_TRAINING_COST_CALCULATOR.md` - Analyze training costs and timing
- `docs/TRAINING_SCRIPTS_README.md` - Detailed training script documentation
- `experiments/TINY_EXPERIMENTS_README.md` - Experiment design details
- `experiments/*/TRAINING_*_CONFIG.md` - Experiment-specific configuration options

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests to help improve this project.

## Citation

If you use this project in your research, please cite the BitNet paper:

```bibtex
@article{ma2023bitnet,
  title={BitNet: Scaling Bitwise Operations for Efficient Transformer Inference and Learning},
  author={Ma, Shuming and Wang, Hongyu and Ma, Lingxiao and Wang, Lei and Wang, Shaoyu and Zhu, Chengyue and Tan, Minghao},
  journal={arXiv preprint arXiv:2310.11453},
  year={2023}
}
```

## Acknowledgments

Built with [PyTorch](https://pytorch.org/) and based on the BitNet research paper on efficient neural network quantization.
