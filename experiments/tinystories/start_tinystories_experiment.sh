#!/bin/bash
#
# TinyStories natural language validation experiment
# Expected runtime: ~2 hours total (10 min tokenizer + 1.5 hrs training)
# Purpose: Verify BitNet works on natural language with tiny model
#

set -e  # Exit on error

# Change to project root
cd "$(dirname "$0")/../.."

echo "==========================================="
echo "BitNet TinyStories Validation Experiment"
echo "==========================================="
echo ""
echo "This will:"
echo "1. Train a custom 2K vocabulary tokenizer"
echo "2. Train a 12M parameter model on simple"
echo "   children's stories"
echo ""
echo "Expected runtime: ~2 hours total"
echo "Expected result: Coherent simple stories"
echo ""
echo "If this works → infrastructure is good!"
echo "If this fails → debug BitNet implementation"
echo ""
echo "==========================================="
echo ""

# Check for required dependencies
echo "Checking dependencies..."
python3 -c "import torch" 2>/dev/null || {
    echo "Error: PyTorch not installed"
    exit 1
}

python3 -c "import datasets" 2>/dev/null || {
    echo "Error: datasets not installed (pip install datasets)"
    exit 1
}

python3 -c "import tokenizers" 2>/dev/null || {
    echo "Error: tokenizers not installed (pip install tokenizers)"
    exit 1
}

python3 -c "import tqdm" 2>/dev/null || {
    echo "Error: tqdm not installed (pip install tqdm)"
    exit 1
}

echo "✓ All dependencies found"
echo ""

# Step 1: Train tokenizer (if not already present)
if [ -f "experiments/tinystories/tinystories_tokenizer.json" ]; then
    echo "✓ Tokenizer already exists: tinystories_tokenizer.json"
    echo ""
else
    echo "Step 1/2: Training tokenizer..."
    echo "Expected time: ~10 minutes"
    echo ""

    python3 experiments/tinystories/train_tinystories_tokenizer.py \
        --vocab-size 2048 \
        --num-samples 100000 \
        --output experiments/tinystories/tinystories_tokenizer.json

    echo ""
    echo "✓ Tokenizer training complete"
    echo ""
fi

# Step 2: Train model
echo "Step 2/2: Training model..."
echo "Expected time: ~1.5 hours"
echo ""

python3 experiments/tinystories/train_bitnet_tiny.py \
    --tokenizer experiments/tinystories/tinystories_tokenizer.json \
    --num-steps 30000 \
    --batch-size 32 \
    --seq-len 256 \
    --log-interval 100 \
    --sample-interval 1000 \
    --checkpoint-interval 5000

echo ""
echo "==========================================="
echo "Experiment complete!"
echo ""
echo "Check results:"
echo "  - Loss curve should drop from ~7 to ~1.5-2.0"
echo "  - Generated stories should be coherent"
echo ""
echo "If successful:"
echo "  → BitNet training infrastructure works!"
echo "  → Problem with 95M model is likely:"
echo "     - Vocabulary too large (32K vs 2K)"
echo "     - Dataset too complex"
echo "     - Needs more training time"
echo ""
echo "Next steps:"
echo "  - Try 95M model with TinyStories dataset"
echo "  - Or train 95M for 10× longer"
echo "  - Or scale to 200M+ parameters"
echo "==========================================="
