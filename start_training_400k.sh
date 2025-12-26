#!/bin/bash
# Launch 400k step BitNet training
# Runtime: ~28 hours (~1.2 days)
# Total tokens: 13.1B
# Checkpoints: Every 10k steps (40 total)

echo "========================================="
echo "BitNet 95M - 400k Step Training"
echo "========================================="
echo ""
echo "Configuration (Optimized for 12GB GPU):"
echo "  Steps:        400,000"
echo "  Batch size:   16 per device"
echo "  Sequence len: 256 tokens"
echo "  Grad accum:   2 steps"
echo "  Effective:    32 (same as original plan)"
echo "  Tokens:       3.3 billion"
echo "  Runtime:      ~77 hours (3.2 days)"
echo "  Checkpoints:  Every 10,000 steps"
echo "  Samples:      Every 5,000 steps"
echo "  Evaluation:   Disabled (--no-eval)"
echo ""
echo "GPU Memory:    Optimized for RTX 3060 12GB"
echo "Context:       256 tokens ≈ 1-2 paragraphs (short-form text)"
echo "Disk space:    ~18-20 GB required"
echo "Buffer:        ~2.8 days remaining after training"
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

# Launch training optimized for 12GB GPU (RTX 3060)
# Batch size = 16 * 2 grad_accum = 32 effective batch size
# Best balance of speed and memory for 400k steps in ~4.6 days
python train_bitnet_95m.py \
    --run-id "${RUN_ID}" \
    --batch-size 16 \
    --seq-len 256 \
    --grad-accum-steps 2 \
    --no-eval

echo ""
echo "========================================="
echo "Training complete!"
echo "Run ID: ${RUN_ID}"
echo "Results: runs/${RUN_ID}/"
echo "========================================="
