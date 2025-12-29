# BitNet 95M - 400K Step Training Configuration

**Configuration finalized:** 2025-12-20
**Training duration:** ~77 hours (3.2 days)
**Total tokens:** 3.3 billion (T/P ≈ 34)

---

## Training Specifications

### Model
- **Parameters:** 95,367,936 (95.4M)
- **Architecture:** BitNet b1.58 with tied embeddings
- **Vocab:** 32,000 (LLaMA-2 tokenizer)
- **Layers:** 8
- **Hidden size:** 768
- **Heads:** 12 query, 6 KV (GQA 2:1)
- **FFN:** 3,072 (SwiGLU)

### Training Hyperparameters (Updated for 12GB GPU)
- **Total steps:** 400,000
- **Batch size:** 16 (per device)
- **Gradient accumulation:** 2 steps
- **Effective batch size:** 32
- **Sequence length:** 256 (optimized for GPU memory)
- **Tokens per step:** 8,192
- **Total tokens:** 3,276,800,000 (~3.3B)
- **T/P ratio:** ~34 (reasonable for compute budget)
- **Note:** Shorter sequences (256 tokens ≈ 1-2 paragraphs) prioritize training speed on limited GPU memory

### Dataset
- **Name:** HuggingFaceFW/fineweb-edu (sample-10BT)
- **Available tokens:** ~10 billion
- **Repetition:** 0.33x (seeing data 0.33 times on average - no repetition)
- **Quality:** High-quality educational web content
- **Context:** 256 tokens per sample (good for short-form text generation)

### Learning Rate Schedule
- **Warmup:** 375 steps (0 → 0.0015)
- **Stage 1 (0-200k):** Linear decay 0.0015 → 0.001, WD=0.1
- **Stage 2 (200k-400k):** Linear decay 0.001 → 0.000015, WD=0.0
- **Stage boundary:** Step 200,000 (mandatory checkpoints at 199,999 and 200,000)

### Logging & Checkpointing
- **Scalar metrics:** Every 100 steps
- **BitNet metrics:** Every 100 steps
- **Evaluation:** Every 5,000 steps (optional, can disable with --no-eval)
- **Samples:** Every 5,000 steps (3 modes × 5 prompts)
- **Checkpoints:** Every 10,000 steps (40 total checkpoints)
- **Mandatory checkpoints:** Steps 199,999 and 200,000 (stage boundary)

---

## Expected Performance

### Runtime (Updated for 12GB GPU Configuration)
- **Steps:** 400,000
- **Step time:** ~695ms (measured with seq_len=256, batch=16, grad_accum=2)
- **Total time:** 400,000 × 0.695s = 278,000s ≈ **77 hours**
- **With overhead:** ~77 hours
- **Completion:** ~3.2 days

### Throughput (Updated for 12GB GPU Configuration)
- **Tokens/sec:** ~11,800 (measured with shorter sequences)
- **GPU utilization:** ~2.2 GB allocated (fits comfortably in 12GB RTX 3060)
- **Memory optimization:** Reduced seq_len from 1024→256 to fit in 12GB VRAM

### Storage
- **Checkpoints:** 40 × 380MB = ~15 GB
- **Metrics:** ~500 MB (400k lines)
- **Samples:** ~3 GB (80 sample events × 5 prompts × 3 modes)
- **Total:** ~18-20 GB

---

## Monitoring Plan

### Automatic
- Progress bar with loss, LR, WD, stage
- Metrics logged to `runs/<run_id>/metrics/scalars.jsonl`
- Samples logged to `runs/<run_id>/samples/samples.jsonl`

### Manual Checks (2-3x per day)
```bash
python check_training_status.py runs/<run_id>
```

**What to watch for:**
- ✅ Loss steadily decreasing (should drop from ~382 to much lower)
- ✅ Samples improving (less repetition, more coherent text)
- ✅ No anomalies (NaN/Inf detection)
- ✅ Throughput stable (~11,800 tokens/sec)
- ⚠️ Loss plateau (may indicate early stopping needed)

### Key Milestones
- **Step 50,000:** ~9.6 hours - First major checkpoint
- **Step 100,000:** ~19.2 hours - Quarter complete
- **Step 199,999:** ~38.5 hours - Pre-stage-transition (mandatory)
- **Step 200,000:** ~38.5 hours - Post-stage-transition (mandatory, WD drops to 0.0)
- **Step 300,000:** ~57.7 hours - Three-quarter complete
- **Step 400,000:** ~77 hours - Final checkpoint

---

## Launching Training

### Default (400k steps with all instrumentation)
```bash
python train_bitnet_95m.py
```

### Without evaluation (faster, recommended for long runs)
```bash
python train_bitnet_95m.py --no-eval
```

### Custom run ID
```bash
python train_bitnet_95m.py --run-id my_400k_run
```

### Resume from checkpoint
```bash
python train_bitnet_95m.py \
    --resume-from runs/old_run/checkpoints/step_100000/checkpoint.pt \
    --run-id my_400k_run_resumed
```

---

## Monitoring During Training

### Check status (from another terminal)
```bash
# Basic status
python check_training_status.py runs/<run_id>

# Watch mode (refresh every 10s)
python check_training_status.py runs/<run_id> --watch
```

### View recent samples
```bash
tail -20 runs/<run_id>/samples/samples.jsonl | jq '.generated_text'
```

### Plot loss curve
```bash
cat runs/<run_id>/metrics/scalars.jsonl | \
    jq -r 'select(.loss != null) | [.step, .loss] | @csv' | \
    python -c "
import sys
import matplotlib.pyplot as plt
data = [line.strip().split(',') for line in sys.stdin]
steps = [float(x[0]) for x in data]
loss = [float(x[1]) for x in data]
plt.plot(steps, loss)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_curve.png')
print('Saved to loss_curve.png')
"
```

---

## Expected Loss Trajectory

Based on typical language model training with seq_len=256:

- **Steps 0-10k:** Loss drops rapidly from ~382 to ~6-7
- **Steps 10k-50k:** Steady decrease ~6 → ~3.5-4
- **Steps 50k-200k:** Gradual improvement ~3.5 → ~2.8-3.2
- **Step 200k:** Stage 2 begins (WD=0.0), may see small bump or acceleration
- **Steps 200k-400k:** Continued improvement ~2.8 → ~2.3-2.7
- **Final (400k):** Expected final loss ~2.3-2.7 (shorter context = slightly higher loss than long-context models, but still good for short-form text)

*Note: Shorter sequences typically result in slightly higher loss values, but model quality for short-form generation will be good.*

---

## When to Stop Early

Consider stopping if:
- Loss plateaus for >50k steps (no improvement)
- Samples stop improving in quality
- Anomalies detected (repeated NaN/Inf)
- Repetition increases in samples (overfitting)

You can always resume later if needed!

---

## Comparison to Configurations

| Metric | Original Plan | 12GB GPU Config | Notes |
|--------|---------------|-----------------|-------|
| Steps | 400,000 | 400,000 | Same |
| Batch size | 32 | 16 (+ grad_accum 2) | Effective batch same |
| Seq length | 1,024 | 256 | 4× reduction for memory |
| Tokens | 13.1B | 3.3B | 4× fewer (shorter contexts) |
| T/P ratio | 137 | 34 | More compute-efficient |
| Runtime | ~28 hours | ~77 hours | Memory-constrained GPU |
| Data repetition | 1.3× | 0.33× | No repetition needed |
| GPU memory | Requires >16GB | Fits in 12GB | RTX 3060 compatible |
| Use case | Long-form text | Short-form text | 1-2 paragraph generation |

**Trade-off:** Shorter context (256 vs 1024 tokens) for GPU memory compatibility. Model will excel at short-form text generation.

---

## Buffer Time Remaining

- **Training duration:** ~77 hours (~3.2 days)
- **Total available:** 144 hours (6 days)
- **Buffer remaining:** ~67 hours (~2.8 days)

This buffer allows for:
- Unexpected slowdowns or issues
- Analysis and evaluation time
- Potential second run with adjusted hyperparameters
- Model evaluation and generation testing

---

## Ready to Launch!

All configuration verified and tested. To start training:

```bash
python train_bitnet_95m.py --no-eval
```

Monitor progress:
```bash
python check_training_status.py runs/<run_id>
```

**Good luck! 🚀**
