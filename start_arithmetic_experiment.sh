#!/bin/bash
#
# Quick arithmetic validation experiment
# Expected runtime: ~30 minutes on RTX 3060
# Purpose: Verify BitNet quantization and gradients work
#

set -e  # Exit on error

echo "========================================"
echo "BitNet Arithmetic Validation Experiment"
echo "========================================"
echo ""
echo "This will train a 5M parameter model on"
echo "synthetic arithmetic (2+3=5 patterns)"
echo ""
echo "Expected runtime: ~30 minutes"
echo "Expected result: Model learns arithmetic"
echo ""
echo "If this fails → infrastructure broken"
echo "If this works → move to TinyStories"
echo ""
echo "========================================"
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

python3 train_bitnet_arithmetic.py \
    --num-steps 10000 \
    --batch-size 32 \
    --seq-len 64 \
    --log-interval 100 \
    --sample-interval 500 \
    --checkpoint-interval 2000

echo ""
echo "========================================"
echo "Experiment complete!"
echo ""
echo "Check results:"
echo "  - Loss curve should drop from ~3.5 to ~0.5"
echo "  - Final test should show correct arithmetic"
echo ""
echo "Next step:"
echo "  ./start_tinystories_experiment.sh"
echo "========================================"
