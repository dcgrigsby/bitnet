#!/bin/bash
#
# Tic-tac-toe game learning experiment
# Expected runtime: ~60 minutes on RTX 3060
# Purpose: Verify BitNet can learn game rules and strategy from synthetic data
#

set -e  # Exit on error

# Change to project root
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "BitNet Tic-Tac-Toe Learning Experiment"
echo "=========================================="
echo ""
echo "This will train a 12M parameter model to"
echo "play tic-tac-toe from random game data"
echo ""
echo "Expected runtime: ~60 minutes"
echo "Expected result: Model learns valid moves"
echo "                 and game completion"
echo ""
echo "If this works → BitNet can learn game rules!"
echo "If this fails → Check game generation logic"
echo ""
echo "=========================================="
echo ""

# Check for required dependencies
echo "Checking dependencies..."
python3 -c "import torch" 2>/dev/null || {
    echo "Error: PyTorch not installed"
    exit 1
}

python3 -c "import numpy" 2>/dev/null || {
    echo "Error: NumPy not installed"
    exit 1
}

python3 -c "import tqdm" 2>/dev/null || {
    echo "Error: tqdm not installed (pip install tqdm)"
    exit 1
}

echo "✓ All dependencies found"
echo ""

# Run training
echo "Starting training..."
echo ""

python3 experiments/tictactoe/train_bitnet_tictactoe.py \
    --num-steps 15000 \
    --batch-size 32 \
    --seq-len 128 \
    --log-interval 100 \
    --sample-interval 500 \
    --checkpoint-interval 2000

echo ""
echo "=========================================="
echo "Experiment complete!"
echo ""
echo "Check results:"
echo "  - Loss curve should drop from ~3-4 to ~0.5-1.0"
echo "  - Model should generate valid game moves"
echo "  - Games should reach completion"
echo ""
echo "Next step:"
echo "  Play against it with:"
echo "  python experiments/tictactoe/play_tictactoe.py <checkpoint_path>"
echo "=========================================="
