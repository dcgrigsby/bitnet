#!/bin/bash
set -e

# --- BitNet 95M DDP Training on 8x RTX 6000 Ada ---
# Launch script for distributed training with torchrun

# --- Paths and Directories ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="bitnet_95M_ddp_${TIMESTAMP}"

# --- Distributed Settings ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export TORCH_DISTRIBUTED_TIMEOUT=3600
NUM_GPUS=8

# --- torch.compile Caching (optional, for future use) ---
export TORCHINDUCTOR_CACHE_DIR="runs/${RUN_ID}/.inductor_cache"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" 2>/dev/null || true

# --- Configuration ---
# Per-GPU batch size: 16 fits in 48GB VRAM
# Global batch size: 16 * 8 = 128
BATCH_SIZE=16
SEQ_LEN=1024
NUM_STEPS=${1:-1000}  # Default 1000 steps, override with first argument

echo "=== BitNet 95M DDP Training ==="
echo "GPUs: ${NUM_GPUS}"
echo "Per-GPU batch size: ${BATCH_SIZE}"
echo "Global batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Sequence length: ${SEQ_LEN}"
echo "Training steps: ${NUM_STEPS}"
echo "Tokens per step: $((BATCH_SIZE * NUM_GPUS * SEQ_LEN))"
echo "Total tokens: $((NUM_STEPS * BATCH_SIZE * NUM_GPUS * SEQ_LEN))"
echo "Run ID: ${RUN_ID}"
echo "================================"
echo ""

# Activate environment
eval "$(~/.local/bin/micromamba shell hook -s bash)" 2>/dev/null || true
micromamba activate bitnet 2>/dev/null || true

# Launch training
torchrun --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_bitnet_95m.py \
  --ddp \
  --batch-size $BATCH_SIZE \
  --seq-len $SEQ_LEN \
  --num-steps $NUM_STEPS \
  --log-interval 50 \
  --checkpoint-interval 5000 \
  --eval-interval 2500 \
  --sample-interval 2500 \
  --run-id "$RUN_ID" \
  --no-eval

echo ""
echo "Training complete: ${RUN_ID}"
echo "Output directory: runs/${RUN_ID}/"
