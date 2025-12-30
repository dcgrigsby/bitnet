# BitNet Training

LLM training experiments with BitNet 1.58-bit quantization.

## Quick Start

### Run Full Experiments (with default settings)

```bash
just exp-arithmetic     # 5M param arithmetic validation (~30 min)
just exp-tictactoe      # 12M param tic-tac-toe game learning (~60 min)
just exp-tinystories    # 12M param natural language (~2 hours)
just exp-baseline       # 95M param baseline (~3-4 days)
```

### Run Training with Custom Arguments

```bash
# Arithmetic experiment with custom steps
just train-arithmetic --num-steps 5000 --batch-size 64

# Tic-tac-toe with custom config
just train-tictactoe --num-steps 10000 --batch-size 64

# TinyStories with custom config
just train-tinystories --num-steps 15000 --seq-len 128 --batch-size 16

# Baseline with custom run-id
just train-baseline --run-id my_experiment --batch-size 32
```

### Play Tic-Tac-Toe

```bash
# Play against trained model
just play-tictactoe runs/bitnet_12M_tictactoe_<timestamp>/checkpoints/step_015000/checkpoint.pt
```

### Utilities

```bash
# Plot training loss
just plot-loss bitnet_5M_arithmetic_1766885392

# Check training status
just check-status bitnet_5M_arithmetic_1766885392

# Watch training status (updates every 10s)
just check-status bitnet_5M_arithmetic_1766885392 --watch

# Chat with trained model
just chat runs/bitnet_95M_400k_1766257283/checkpoint_10000.pt
```

## Project Structure

```
bitnet/
├── experiments/          # Organized by experiment
│   ├── arithmetic/       # 5M arithmetic validation
│   ├── tictactoe/        # 12M tic-tac-toe game
│   ├── tinystories/      # 12M natural language
│   └── baseline-95m/     # 95M baseline
├── scripts/              # Utility scripts
├── docs/                 # Documentation & configs
├── runs/                 # Training outputs
└── src/bitnet/          # Core BitNet implementation
```

## Experiments

### Arithmetic (5M params)
- **Purpose**: Verify BitNet quantization works
- **Dataset**: Synthetic arithmetic (2+3=5)
- **Runtime**: ~30 minutes
- **Expected**: Loss drops from ~3.5 to ~0.5

### Tic-Tac-Toe (12M params)
- **Purpose**: Learn game rules and strategy from data
- **Dataset**: Random-play synthetic games
- **Runtime**: ~60 minutes (15k steps)
- **Expected**: Loss drops from ~3-4 to ~0.5-1.0
- **Interactive**: Play against trained model!

### TinyStories (12M params)
- **Purpose**: Verify natural language learning
- **Dataset**: Simple children's stories
- **Runtime**: ~2 hours (10 min tokenizer + 1.5 hrs training)
- **Expected**: Loss drops from ~7 to ~1.5-2.0

### Baseline 95M
- **Purpose**: Full-scale baseline training
- **Dataset**: FineWeb-Edu
- **Runtime**: ~77 hours (400k steps)
- **Config**: 16 batch size, 256 seq len, 2 grad accum

## Development

```bash
# Run tests
just test

# Run specific test
just test tests/test_model.py

# Simple training test
just train-simple 100
```

## Documentation

- `docs/BITNET_TRAINING_COST_CALCULATOR.md` - Training cost analysis
- `docs/TRAINING_SCRIPTS_README.md` - Training script details
- `experiments/TINY_EXPERIMENTS_README.md` - Experiment overview
- `experiments/*/TRAINING_*_CONFIG.md` - Experiment-specific configs

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests to help improve this project.

## Citation

If you use this project in your research, please cite it appropriately. BitNet 1.58-bit quantization research is based on the following:

```bibtex
@article{ma2024bitnet,
  title={BitNet: Scaling Bitwise Operations for Efficient Transformer Inference and Learning},
  author={Ma, Shuming and Wang, Hongyu and Ma, Lingxiao and Wang, Lei and Wang, Shaoyu and Zhu, Chengyue and Tan, Minghao},
  journal={arXiv preprint arXiv:2402.18029},
  year={2024}
}
```

## Acknowledgments

Built with [PyTorch](https://pytorch.org/) and inspired by the BitNet research paper on efficient neural network quantization.
