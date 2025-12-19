# BitNet Scaling Investigation
**Date:** December 18, 2025
**Status:** Testing 4 model sizes with standard BitNet parameters

---

## Hypothesis

The degenerate decoding problem might be a **scaling issue**, not a hyperparameter issue.

**Theory:**
- Microsoft tested BitNet on models from 700M to 3.9B parameters
- 256-hidden model is ~1000x smaller than the minimum
- Standard LR (0.0015) and WD schedule (0.1→0.0) may be calibrated for larger models
- Smaller models might need different hyperparameters OR have hard minimum size

**Test:** Does vocabulary collapse decrease as we increase model size?

---

## Setup

### Models Being Tested

| Size | Hidden | Layers | Heads | FFN | Parameters | vs 256-baseline |
|------|--------|--------|-------|-----|------------|-----------------|
| **256** | 256 | 4 | 4 | 512 | 28.4M | 1.0x (baseline) |
| **512** | 512 | 4 | 4 | 1024 | 114M | 4.0x |
| **768** | 768 | 4 | 4 | 1536 | 257M | 9.0x |
| **1024** | 1024 | 4 | 4 | 2048 | 458M | 16.1x |

Note: Layers, heads, and proportional scaling kept constant.

### Hyperparameters

**All models use STANDARD BitNet parameters from papers:**

```
Learning Rate:
  - Peak: 0.0015 (standard)
  - Warmup: 375 steps
  - Stage 1: Linear decay peak → 2/3 peak
  - Stage 2: Linear decay 2/3 peak → final (0.1% of peak)

Weight Decay:
  - Stage 1: 0.1 (per papers)
  - Stage 2: 0.0 (per papers - CORRECTED from 0.05)

Optimization:
  - Optimizer: Adam(lr=0.0015, betas=(0.9, 0.95))
  - Gradient clipping: norm=1.0
  - Training steps: 10,000 (Stage 1-only, no Stage 2 transition)
```

### Metric

**Unique token predictions** sampled every 1000 steps:
- Generator: 20 random 8-token sequences
- Metric: Count unique predicted next tokens across 20 samples
- Range: 1 (complete collapse) to 50,257 (uniform distribution)
- Healthy: >100 unique tokens

---

## What We're Testing

### Key Question
**"Is there a minimum model size where BitNet training becomes stable?"**

### Expected Outcomes

**Scenario A: Scaling Fixes It (Most Likely)**
- 256-hidden: Collapses to 2-6 tokens ✗
- 512-hidden: Improves, maybe 20-50 tokens ⚠
- 768-hidden: Better, maybe 50-100 tokens ⚠
- 1024-hidden: Stable, >100 tokens ✓
- **Conclusion:** BitNet has minimum viable model size

**Scenario B: LR is the Problem (Less Likely)**
- All models collapse similarly regardless of size
- **Conclusion:** LR needs adjustment for small models

**Scenario C: Something Else**
- Unexpected pattern, triggers deeper investigation

---

## Why This Matters

### If Scaling Fixes It
1. 256-hidden is **below BitNet's design envelope**
2. Need models of 512+ hidden for stable training
3. Microsoft's papers focused on 700M+ for good reason
4. Could document minimum viable model size

### If Scaling Doesn't Fix It
1. Problem is NOT just model size
2. LR (0.0015) likely too aggressive for small models
3. Need to adjust LR based on model scale
4. Creates precedent for adapting BitNet to smaller models

### Practical Impact
- If scaling fixes it: Use 512+ hidden models
- If LR needs adjustment: Scale LR with model size
- Either way: Understanding the boundary helps future work

---

## Running Tests

### Test Configuration

Each model runs:
```
python examples/test_model_scaling.py 10000 {hidden_size} /tmp/scale_{hidden_size}_10k.txt
```

### Parallel Execution
Testing 4 models in parallel:
- 256-hidden: ID b864d0d
- 512-hidden: ID b4078d4
- 768-hidden: ID b428aea
- 1024-hidden: ID b747399

### Expected Duration
- Per model: ~4-6 minutes on CPU
- Total: ~5-6 minutes parallel (all at once)
- All tests should complete in <10 minutes

---

## Analysis Plan

When tests complete:

### 1. Collect All Results
```
for size in 256 512 768 1024; do
    echo "Hidden $size:"
    tail -5 /tmp/scale_${size}_10k.txt
done
```

### 2. Create Comparison Table
Compare unique token counts at each checkpoint (steps 1k, 4k, 10k)

### 3. Plot Trajectory
See if larger models show more stable vocabulary across training

### 4. Identify Threshold
What minimum hidden_size keeps >100 unique tokens?

### 5. Hypothesis Conclusion
- If 512-hidden works: Model scaling is the issue
- If all collapse: LR magnitude is the issue
- If other pattern: Requires deeper investigation

---

## Expected Results Preview

**Based on earlier finding that reduced LR enables recovery:**

The mechanism might be:
- Larger models have more capacity
- More capacity provides more "escape routes" from collapse attractor
- Standard LR might work fine for larger models
- Smaller models get stuck because they lack capacity

**If this is true:**
- 256-hidden: ~2-6 tokens (confirmed collapser)
- 512-hidden: ~10-30 tokens (partial collapse)
- 768-hidden: ~50-100 tokens (marginal)
- 1024-hidden: ~200+ tokens (healthy)

This would show clear scaling threshold.

---

## Comparison to Earlier Tests

### Earlier Findings (Baseline)
- 256-hidden + standard LR + old WD (0.05): Collapses to 2 tokens ✗
- 256-hidden + reduced LR (0.0005): Recovers to 18 tokens ✓

### Current Test
- 256-hidden + standard LR + corrected WD (0.0): Will show effect of WD fix
- 512-768-1024-hidden + standard LR + corrected WD (0.0): Will show scaling effect

### Key Difference
- Previous test changed LR (0.0015 → 0.0005) on same model
- This test changes model size (256 → 1024) with standard LR
- Separates "LR for small models" from "minimum model size" questions

---

## Files Created This Phase

- `examples/test_model_scaling.py` - Main scaling test script
- `examples/run_scaling_tests.sh` - Bash runner for all sizes
- Results: `/tmp/scale_{256,512,768,1024}_10k.txt` (being generated)
- `SCALING_INVESTIGATION.md` - This document

---

## Next Steps After Results

### If Scaling Threshold Found
1. **Document minimum model size**
2. **Create separate configs for different scales**
3. **Test if reduced LR still helps at larger sizes**

### If No Clear Threshold
1. **Combine findings:** Larger model + reduced LR test
2. **Investigate:** Whether it's capacity vs LR schedule
3. **Propose:** Adaptive LR scaling function

### Regardless
1. **Commit all findings**
2. **Update BitNetConfig documentation**
3. **Provide recommendations for future use**

---

## Summary

This investigation will answer a fundamental question about BitNet training:

**Is the collapse a property of 1-bit training that manifests at small scales,**
**or a property of the hyperparameter set we're using?**

The answer will guide whether we:
- Build larger BitNet models
- Adapt hyperparameters for small models
- Or both with clear scaling laws
