# BitNet 95M - 400K Step Training Configuration

**Configuration finalized:** 2025-12-20
**Training duration:** ~28 hours (~1.2 days)
**Total tokens:** 13.1 billion (T/P ≈ 137)

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

### Training Hyperparameters
- **Total steps:** 400,000
- **Batch size:** 32
- **Sequence length:** 1,024
- **Tokens per step:** 32,768
- **Total tokens:** 13,107,200,000 (~13.1B)
- **T/P ratio:** ~137 (beyond Chinchilla optimal of ~20)

### Dataset
- **Name:** HuggingFaceFW/fineweb-edu (sample-10BT)
- **Available tokens:** ~10 billion
- **Repetition:** 1.3x (seeing data 1.3 times on average)
- **Quality:** High-quality educational web content

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

### Runtime
- **Steps:** 400,000
- **Step time:** ~250ms (measured)
- **Total time:** 400,000 × 0.25s = 100,000s ≈ **27.8 hours**
- **With overhead:** ~28-30 hours
- **Completion:** ~1.2 days

### Throughput
- **Tokens/sec:** ~20,000 (measured in test)
- **GPU utilization:** ~1.7 GB allocated, ~2.6 GB reserved

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
- ✅ Loss steadily decreasing (should drop from ~140 to much lower)
- ✅ Samples improving (less repetition, more coherent text)
- ✅ No anomalies (NaN/Inf detection)
- ✅ Throughput stable (~20k tokens/sec)
- ⚠️ Loss plateau (may indicate early stopping needed)

### Key Milestones
- **Step 50,000:** ~3.5 hours - First major checkpoint
- **Step 100,000:** ~7 hours - Quarter complete
- **Step 199,999:** ~14 hours - Pre-stage-transition (mandatory)
- **Step 200,000:** ~14 hours - Post-stage-transition (mandatory, WD drops to 0.0)
- **Step 300,000:** ~21 hours - Three-quarter complete
- **Step 400,000:** ~28 hours - Final checkpoint

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

Based on typical language model training:

- **Steps 0-10k:** Loss drops rapidly from ~150 to ~8
- **Steps 10k-50k:** Steady decrease ~8 → ~4
- **Steps 50k-200k:** Gradual improvement ~4 → ~2.5
- **Step 200k:** Stage 2 begins (WD=0.0), may see small bump or acceleration
- **Steps 200k-400k:** Continued improvement ~2.5 → ~1.8-2.0
- **Final (400k):** Expected final loss ~1.8-2.2 (will be much better than 60k run)

*Note: These are rough estimates. Actual values will vary.*

---

## When to Stop Early

Consider stopping if:
- Loss plateaus for >50k steps (no improvement)
- Samples stop improving in quality
- Anomalies detected (repeated NaN/Inf)
- Repetition increases in samples (overfitting)

You can always resume later if needed!

---

## Comparison to Chinchilla Optimal

| Metric | Chinchilla (60k) | Extended (400k) | Improvement |
|--------|------------------|-----------------|-------------|
| Steps | 60,000 | 400,000 | 6.7× |
| Tokens | 1.97B | 13.1B | 6.7× |
| T/P ratio | 20.6 | 137 | 6.7× |
| Runtime | ~4 hours | ~28 hours | 7× |
| Data repetition | 0.2× | 1.3× | 6.5× |
| Expected final loss | ~3.5-4.0 | ~1.8-2.2 | ~40-50% better |

**Trade-off:** Uses 7× more compute for ~40-50% better loss (diminishing returns, but worthwhile given available time).

---

## Buffer Time Remaining

- **Training duration:** ~28 hours
- **Total available:** 144 hours (6 days)
- **Buffer remaining:** ~116 hours (4.8 days)

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
