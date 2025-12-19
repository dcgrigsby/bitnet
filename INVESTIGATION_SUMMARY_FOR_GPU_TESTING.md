# BitNet Degenerate Decoding Investigation Summary

**For GPU Verification Testing**

---

## The Problem

Model collapses to 2-6 unique tokens by end of training. Expected: >1000 unique tokens.

### Evidence of Problem
From `FINAL_REPORT.txt`:
- 1000-cycle run: Collapsed to 1-8 tokens by end
- 10000-cycle run: Collapsed to ~5 tokens by end
- 20000-cycle run: Collapsed to 2-8 tokens by end
- **Pattern:** All runs collapse severely, regardless of training length

---

## Investigation Process (Phases 1-5)

### Phase 1: Root Cause Investigation ✅
**Finding:** Collapse occurs in Stage 1 itself (steps 2-4k), not at transition

**Evidence:** `train_stage1_only.py` diagnostic
```
Step 1:    20 tokens
Step 1k:   19 tokens
Step 2k:    9 tokens
Step 3k:    4 tokens
Step 4k:    2 tokens (COLLAPSED)
Step 7k:    2 tokens (STUCK)
```

### Phase 2: Pattern Analysis ✅
**Finding:** Loss IMPROVES while tokens collapse

**Evidence:** Loss 10.99→10.25 while tokens 20→2

Proves CrossEntropyLoss naturally rewards mode collapse

### Phase 3: Hypothesis Testing ✅
**Hypothesis:** Reduced LR enables recovery

**Test:** `train_reduced_lr.py` with LR=0.0005 (vs standard 0.0015)

**Result:**
```
Step 1k:    3 tokens (collapsed faster)
Step 2k:   10 tokens (RECOVERED!)
Step 3k:   14 tokens
Step 4k:   18 tokens
```

**Conclusion:** Model CAN achieve diversity with right conditions

### Phase 4: Solution Testing ✅
Tested other hypotheses - all failed:
- Label smoothing: No effect ❌
- Gradient clipping: Made it worse ❌
- Only reduced LR worked ✅

### Phase 5: Scaling Investigation + Discovery ✅

**Setup:** Test 4 model sizes (256, 512, 768, 1024 hidden) with:
- LR: 0.0015 (standard from papers)
- **WD: 0.0 in Stage 2** (corrected from 0.05)

**CRITICAL DISCOVERY:**

All 4 models showed identical recovery pattern:
```
Stage 1 (WD=0.1):   20 → 2 tokens (collapse)
Stage 2 (WD=0.0):   2 → 11-13 tokens (RECOVERY!)
```

### The Fix
Changed one line in `src/bitnet/train.py`:
```python
# BEFORE (WRONG):
self.stage2_wd: float = 0.05

# AFTER (CORRECT):
self.stage2_wd: float = 0.0
```

This matches Microsoft's official BitNet training specification.

---

## Current Evidence (CPU Tests, 10k Steps)

### What We Know
✅ WD=0.0 in Stage 2 enables recovery to 11-13 tokens
✅ Recovery happens consistently across all model sizes
✅ Recovery correlates with WD schedule change at step 5000
✅ Loss drops ~29% when WD changes (7.32 from 10.34)

### What We DON'T Know
❌ Whether 11-13 tokens is enough (need >50-100 for healthy diversity)
❌ Whether model stays diverse through full Stage 2 training
❌ Whether it collapses again later in training (step 10k-20k)
❌ Whether it actually generates good text

### The Gap
- **Previous testing (FINAL_REPORT):** Up to 20k steps, showed eventual collapse
- **My testing:** Only 10k steps (just into Stage 2), shows recovery but too short
- **Needed:** 20k+ step tests to prove end-of-training stability

---

## Testing Scripts Available

### Main Test Script
**File:** `examples/test_model_scaling.py`

**Usage:**
```bash
python examples/test_model_scaling.py <num_steps> <hidden_size> <output_file>
```

**Parameters:**
- `num_steps`: How many training steps (20000 for full verification)
- `hidden_size`: Model size (256, 512, 768, or 1024)
- `output_file`: Where to save results

**Output:** CSV with columns:
- Step
- Loss
- UniquePredictions (token count - THIS IS THE KEY METRIC)
- LR (learning rate)
- WD (weight decay)

### Other Diagnostic Scripts
- `train_stage1_only.py` - Stage 1 only (already tested)
- `train_reduced_lr.py` - Reduced LR testing (already tested)
- `train_with_label_smoothing.py` - Loss function testing (already tested)
- `train_aggressive_grad_clip.py` - Gradient testing (already tested)

---

## What Will Prove the Fix Works

### Minimum Evidence (20k-step test, 256-hidden)
```
Step 20k: >50 unique tokens
```

If true: Fix enables enough diversity for real training ✅

### Strong Evidence (20k-step tests, multiple sizes)
```
256-hidden step 20k: >50 tokens
512-hidden step 20k: >50 tokens
```

If true: Fix works consistently across scales ✅

### Comprehensive Evidence (30k-step extended test)
```
768-hidden step 30k: >50 tokens
```

If true: Model stays stable through extended training ✅

---

## What Will Show the Fix Doesn't Work

### If 256-hidden step 20k: <10 tokens
This means:
- Recovery at step 10k is temporary
- Collapses again later in Stage 2
- WD schedule alone isn't the solution
- Might need to also reduce LR or use larger models

### If pattern is highly volatile
Tokens jumping 2→10→3→15→2 suggests:
- Unstable training dynamics
- Might need additional fixes beyond WD

---

## Key Files to Review

### Configuration (Has the fix)
- `src/bitnet/train.py` - Look at `TwoStageWDScheduler.stage2_wd`

### Original Problem Evidence
- `FINAL_REPORT.txt` - Shows collapse by training end

### Analysis Documents
- `SYSTEMATIC_DEBUGGING_FINAL_REPORT.md` - Phases 1-4 details
- `CRITICAL_FINDING_WD_FIX.md` - WD fix discovery
- `SCALING_TEST_FINAL_RESULTS.md` - All 4 model size results (10k steps)

---

## Hyperparameter Summary

### Standard BitNet (per Microsoft papers)
```
Learning Rate:
  - Peak: 0.0015
  - Warmup: 375 steps
  - Stage boundary: 50% of training

Weight Decay:
  - Stage 1: 0.1
  - Stage 2: 0.0  ← THIS IS THE FIX (was 0.05)

Other:
  - Optimizer: Adam (beta1=0.9, beta2=0.95)
  - Gradient clip: 1.0
```

---

## Timeline Comparison

### Original Problem (CPU, 20k steps)
- ~20 minutes to train and collapse

### Current Testing (CPU, 10k steps)
- ~5-10 minutes to show recovery

### GPU Verification (20k steps)
- ~2 minutes to train
- ~15 minutes total with all tests

---

## Success Criteria

**The fix is proven to work if:**
1. 256-hidden trains 20k steps and ends with >50 tokens
2. 512-hidden trains 20k steps and ends with >50 tokens
3. Pattern shows recovery at step 6k (WD change) and stays stable through step 20k

**The fix is proven to NOT work if:**
1. Any model ends at <10 tokens after 20k steps
2. Recovery is temporary (high at 10k, low at 20k)
3. Training crashes or shows erratic behavior

---

## Questions This Will Answer

1. **Is WD schedule the root cause?** Yes if >50 tokens at step 20k
2. **Can 256-hidden model work?** Yes if it stays >50 at end
3. **Does it scale?** Yes if multiple sizes show same behavior
4. **Is additional LR adjustment needed?** No if this works; Yes if still collapses

---

## Next Steps After GPU Testing

### If Fix Works (>50 tokens at 20k)
1. Run full training on real dataset
2. Test actual text generation quality
3. Document as solved
4. Update training guidelines

### If Fix Doesn't Work (<10 tokens at 20k)
1. Combine approaches: WD=0.0 + reduced LR=0.0005
2. Or test: larger models with standard LR
3. Continue investigation

---

## Important Notes for GPU Testing

1. **Don't panic if step 5k shows 2 tokens** - that's expected (Stage 1 collapse)
2. **Focus on step 20k final token count** - that's the verdict
3. **Loss will drop at step 6k transition** - that's expected and good
4. **Variance in token counts is normal** - look at the overall trend

---

## Contact/Handoff Info

When you have a GPU system ready, you need:

1. Copy the files listed in GPU_VERIFICATION_PLAN.md
2. Run the test commands in the order specified
3. Send back the output CSV files
4. Report the step 20k token counts for each model

That's all needed to definitively answer whether the fix works.
