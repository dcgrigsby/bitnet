#!/bin/bash
# Test BitNet training across multiple model sizes
# This tests the hypothesis: collapse is due to model being too small

echo "=========================================="
echo "BitNet Scaling Test Suite"
echo "=========================================="
echo "Testing model sizes: 256, 512, 768, 1024 hidden"
echo "Using standard BitNet LR: 0.0015"
echo "Using corrected WD: 0.1 → 0.0"
echo ""

NUM_STEPS=10000

echo "Starting tests... (this will take a while)"
echo ""

# Test each model size
for hidden_size in 256 512 768 1024; do
    echo "Testing hidden_size=${hidden_size}..."
    python examples/test_model_scaling.py ${NUM_STEPS} ${hidden_size} /tmp/scale_${hidden_size}_${NUM_STEPS}.txt
    echo ""
done

echo "=========================================="
echo "All tests complete!"
echo "=========================================="
echo ""
echo "Results:"
for hidden_size in 256 512 768 1024; do
    echo ""
    echo "--- hidden_size=${hidden_size} ---"
    tail -5 /tmp/scale_${hidden_size}_${NUM_STEPS}.txt
done

echo ""
echo "Summary table:"
echo "Hidden | Step 1k | Step 4k | Step 10k"
echo "--------|---------|---------|----------"
for hidden_size in 256 512 768 1024; do
    step1k=$(grep "^1000," /tmp/scale_${hidden_size}_${NUM_STEPS}.txt | cut -d, -f3)
    step4k=$(grep "^4000," /tmp/scale_${hidden_size}_${NUM_STEPS}.txt | cut -d, -f3)
    step10k=$(grep "^10000," /tmp/scale_${hidden_size}_${NUM_STEPS}.txt | cut -d, -f3)
    echo " ${hidden_size}  |   ${step1k}   |   ${step4k}   |   ${step10k}"
done
