#!/bin/bash
# Launch 400k step BitNet training
# Runtime: ~28 hours (~1.2 days)
# Total tokens: 13.1B
# Checkpoints: Every 10k steps (40 total)

echo "========================================="
echo "BitNet 95M - 400k Step Training"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Steps:        400,000"
echo "  Tokens:       13.1 billion"
echo "  Runtime:      ~28 hours (~1.2 days)"
echo "  Checkpoints:  Every 10,000 steps"
echo "  Samples:      Every 5,000 steps"
echo "  Evaluation:   Disabled (--no-eval)"
echo ""
echo "Disk space required: ~18-20 GB"
echo "Buffer remaining: ~4.8 days"
echo ""
echo "========================================="
echo ""

# Generate run ID with timestamp
TIMESTAMP=$(date +%s)
RUN_ID="bitnet_95M_400k_${TIMESTAMP}"

echo "Run ID: ${RUN_ID}"
echo ""
echo "Monitor progress with:"
echo "  python check_training_status.py runs/${RUN_ID}"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

# Launch training
python train_bitnet_95m.py \
    --run-id "${RUN_ID}" \
    --no-eval

echo ""
echo "========================================="
echo "Training complete!"
echo "Run ID: ${RUN_ID}"
echo "Results: runs/${RUN_ID}/"
echo "========================================="
