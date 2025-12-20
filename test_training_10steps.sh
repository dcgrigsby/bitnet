#!/bin/bash
# Quick 10-step test to verify training infrastructure

echo "Running 10-step training test..."
echo "This will verify:"
echo "  - Data loading works"
echo "  - Model forward/backward pass works"
echo "  - Instrumentation writes files correctly"
echo "  - Status checker can read the files"
echo ""

python train_bitnet_95m.py \
    --run-id test_10steps \
    --num-steps 10 \
    --batch-size 4 \
    --seq-len 128 \
    --log-interval 1 \
    --checkpoint-interval 5 \
    --sample-interval 5 \
    --eval-interval 5 \
    --no-eval

echo ""
echo "Training test complete. Checking status..."
echo ""

python check_training_status.py runs/test_10steps

echo ""
echo "Verifying directory structure..."
ls -lh runs/test_10steps/
ls -lh runs/test_10steps/meta/
ls -lh runs/test_10steps/checkpoints/
ls -lh runs/test_10steps/metrics/
ls -lh runs/test_10steps/samples/

echo ""
echo "Test complete!"
