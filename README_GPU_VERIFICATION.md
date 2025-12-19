# BitNet Degenerate Decoding Fix - GPU Verification

**Status:** Fix identified (WD=0.0 in Stage 2), needs GPU verification

**What to do:** Run the verification plan on a system with GPU

---

## Quick Start

### If you're about to run this on GPU:

1. **Read first:**
   - `GPU_VERIFICATION_PLAN.md` - Step-by-step what to run

2. **Copy to GPU system:**
   - `src/bitnet/` - All source files
   - `examples/test_model_scaling.py` - Main test script
   - `examples/train_simple.py` - Reference

3. **Run in order:**
   ```bash
   python examples/test_model_scaling.py 20000 256 /tmp/gpu_verify_256_20k.txt
   python examples/test_model_scaling.py 20000 512 /tmp/gpu_verify_512_20k.txt
   ```

4. **Send back:**
   - The .txt output files
   - Report of final token counts at step 20k

---

## What's the Problem?

Model collapses to 2-6 tokens. Should be >100+ tokens.

```
Current (broken): 20 tokens → 2 tokens (COLLAPSE)
Expected (fixed):  20 tokens → 2 tokens → 11-13 tokens (RECOVERY)
Ideal (unknown):   20 tokens → diverse tokens throughout training
```

---

## What's the Hypothesis?

Using WD=0.05 in Stage 2 (WRONG) instead of WD=0.0 (CORRECT per papers) prevents model from escaping collapse.

**Fix:** Changed one line in `src/bitnet/train.py`:
```python
self.stage2_wd: float = 0.0  # Was 0.05
```

---

## What Evidence Do We Have?

### On CPU (10k steps):
✅ Shows recovery to 11-13 tokens when WD changes to 0.0
❌ But doesn't test long enough through Stage 2

### What We Need:
✅ 20k-step tests on GPU (fast enough to verify)
✅ Check if model stays diverse through full Stage 2
✅ Check final token count at training end

---

## Key Documents

**To Understand the Problem:**
- `INVESTIGATION_SUMMARY_FOR_GPU_TESTING.md` - Full background

**To Run the Tests:**
- `GPU_VERIFICATION_PLAN.md` - Exact commands and what to expect

**Historical Analysis (if interested):**
- `SYSTEMATIC_DEBUGGING_FINAL_REPORT.md` - Phases 1-4 findings
- `SCALING_TEST_FINAL_RESULTS.md` - Current CPU test results
- `FINAL_REPORT.txt` - Original problem evidence

---

## Verification Checklist

### Before Running
- [ ] Have GPU system ready (RTX 3060 or better)
- [ ] Have copied `src/bitnet/` files
- [ ] Have `examples/test_model_scaling.py`
- [ ] Read `GPU_VERIFICATION_PLAN.md`

### While Running
- [ ] Step 5k shows ~2 tokens (expected collapse)
- [ ] Step 6k shows jump to >8 tokens (expected recovery)
- [ ] Step 20k shows token count (CRITICAL)

### After Running
- [ ] Collected output CSV files
- [ ] Noted final token counts
- [ ] Ready to report results

---

## Expected Results

### If Fix Works
```
256-hidden at step 20k: >50 tokens ✅
512-hidden at step 20k: >50 tokens ✅
```
→ Problem is solved

### If Fix Doesn't Work
```
256-hidden at step 20k: <10 tokens ❌
```
→ Need additional changes (reduced LR or larger model)

---

## Time to Run

- Phase 1 (5k steps): 30 seconds
- Phase 2 (20k steps, 256): 2 minutes
- Phase 3 (20k steps, 512): 3 minutes
- **Total: ~6 minutes**

Much faster than the 20+ minutes on CPU.

---

## What This Answers

**Main question:** Does WD=0.0 in Stage 2 solve degenerate decoding?

**Sub-questions:**
1. Can 256-hidden stay diverse through full Stage 2?
2. Is the fix consistent across model sizes?
3. Do we need additional changes?

---

## Files You'll Need to Copy

### Required
```
src/bitnet/config.py
src/bitnet/data.py
src/bitnet/linear.py
src/bitnet/quant.py
src/bitnet/train.py          ← HAS THE FIX
src/bitnet/transformer.py

examples/test_model_scaling.py ← MAIN TEST SCRIPT
```

### Optional (reference)
```
examples/train_simple.py
examples/train_stage1_only.py
examples/train_reduced_lr.py
```

---

## Instructions for Running on GPU System

### Step 1: Set Up
```bash
cd /path/to/bitnet
python examples/test_model_scaling.py 5000 256 /tmp/quick_test.txt
```

If this works, you have GPU access set up correctly.

### Step 2: Run Full Tests
```bash
# 256-hidden: critical test
python examples/test_model_scaling.py 20000 256 /tmp/gpu_verify_256_20k.txt

# 512-hidden: confirmation test
python examples/test_model_scaling.py 20000 512 /tmp/gpu_verify_512_20k.txt
```

### Step 3: Check Results
```bash
tail -5 /tmp/gpu_verify_256_20k.txt
tail -5 /tmp/gpu_verify_512_20k.txt
```

Look at the last line - the "UniquePredictions" column is the token count.

### Step 4: Report Back
Send:
- The .txt files
- Screenshot or copy of final step (showing token counts)
- Whether you think it worked

---

## Common Issues

### "Out of memory"
- Use smaller model (256 instead of 512)
- Or reduce batch size in code

### "Training is still slow"
- Check that GPU is being used (nvidia-smi)
- Make sure you're running on actual GPU, not CPU

### "Got weird token counts (0, 50000+, etc)"
- That's a bug, not a valid result
- Check if training ran correctly
- Look at loss values (should be 6-11 range)

---

## Success Definition

**The fix works if:**
- 256-hidden step 20k shows >50 tokens
- Loss trajectory looks normal (decreases smoothly)
- No errors in training

**The fix doesn't work if:**
- Step 20k shows <10 tokens
- Recovery is temporary (high at 10k, low at 20k)
- Training crashes

---

## Summary for Handoff

You now have:
1. A clear fix (WD=0.0 in Stage 2)
2. Test scripts ready to run
3. A verification plan (GPU_VERIFICATION_PLAN.md)
4. Expected results documented
5. All analysis of how we got here

**To verify on GPU:** Follow GPU_VERIFICATION_PLAN.md, run the commands, send back the results.

**That's all you need to know to verify whether this fix actually solves the problem.**
