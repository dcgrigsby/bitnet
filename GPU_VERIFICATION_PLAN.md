# BitNet WD Fix Verification Plan - GPU Execution

**Purpose:** Verify whether correcting Stage 2 weight decay (WD=0.1→0.0) actually solves the degenerate decoding problem

**Prerequisite:** System with GPU (RTX 3060 or better)

**Expected Duration on GPU:** ~5-10 minutes total

---

## Current Status

### Problem Found
BitNet b1.58 collapses to 2-6 unique tokens by end of training

### Hypothesis
Using WD=0.05 in Stage 2 (wrong) instead of WD=0.0 (correct per papers) prevents recovery

### Evidence So Far
- Stage 1-only tests (10k steps) with WD=0.0 show recovery to 11-13 tokens
- But these tests don't run long enough through Stage 2 to prove end-of-training stability
- Need longer tests to verify fix actually works

### What's Unknown
- Does model maintain diversity through full 20k+ step training?
- Does it collapse again later in Stage 2?
- What are final token counts at training end?

---

## Verification Plan (Step-by-Step)

### Phase 1: Baseline Test (Without GPU Optimization)
**Goal:** Verify the WD=0.0 fix is actually in the code

```bash
python examples/test_model_scaling.py 5000 256 /tmp/verify_baseline_256_5k.txt
```

**Expected result:**
- Collapse to 2 tokens around step 2-4k
- Recovery to 8+ tokens when WD changes to 0.0 at step 2500

**Time:** ~30 seconds on GPU

---

### Phase 2: Full Training Test - 256-hidden (Critical)
**Goal:** Does the small model stay diverse through full Stage 1+2?

```bash
python examples/test_model_scaling.py 20000 256 /tmp/gpu_verify_256_20k.txt
```

**What to check:**
- Step 1k: Should be ~16-20 tokens (healthy start)
- Step 5k: Will collapse to ~2 tokens (normal Stage 1 end)
- Step 6k: Should jump to ~10+ tokens (WD changes, recovery)
- Step 10k: Should be ~10+ tokens (mid Stage 2)
- **Step 20k: CRITICAL - final token count** (should be >50, ideally >100)

**Expected result:**
- If final is >50: Fix works, model recovers and stays diverse
- If final is <10: Fix doesn't work, collapses again later in Stage 2
- If it varies: Might show unstable behavior

**Time:** ~2 minutes on GPU

---

### Phase 3: Full Training Test - 512-hidden
**Goal:** Confirm larger model also stays diverse

```bash
python examples/test_model_scaling.py 20000 512 /tmp/gpu_verify_512_20k.txt
```

**What to check:** Same as 256, but check step 20k final tokens

**Expected result:** Should match or exceed 256-hidden performance

**Time:** ~3 minutes on GPU

---

### Phase 4: Extended Training Test - 768-hidden (Optional but Recommended)
**Goal:** Test even longer training trajectory

```bash
python examples/test_model_scaling.py 30000 768 /tmp/gpu_verify_768_30k.txt
```

**What to check:**
- Steps 5k, 10k, 15k, 20k, 25k, 30k
- Look for patterns: Does it stay stable or degrade?
- Final token count at 30k

**Time:** ~5 minutes on GPU

---

## How to Run This

### On the GPU System

1. **Copy these files:**
   ```
   src/bitnet/
   examples/test_model_scaling.py
   examples/train_simple.py (if it exists, for reference)
   ```

2. **Run the plan in order:**
   ```bash
   # Phase 1 - quick verification
   python examples/test_model_scaling.py 5000 256 /tmp/verify_baseline_256_5k.txt

   # Phase 2 - critical test
   python examples/test_model_scaling.py 20000 256 /tmp/gpu_verify_256_20k.txt

   # Phase 3 - confirmation test
   python examples/test_model_scaling.py 20000 512 /tmp/gpu_verify_512_20k.txt

   # Phase 4 - extended test (optional)
   python examples/test_model_scaling.py 30000 768 /tmp/gpu_verify_768_30k.txt
   ```

3. **Collect outputs:**
   - `/tmp/verify_baseline_256_5k.txt`
   - `/tmp/gpu_verify_256_20k.txt`
   - `/tmp/gpu_verify_512_20k.txt`
   - `/tmp/gpu_verify_768_30k.txt` (if run)

4. **Send back the output files and report**

---

## How to Interpret Results

### Success Criteria (Fix Works)

**All of these must be true:**
1. Step 1k: >10 tokens (healthy)
2. Step 5k: ~2 tokens (expected collapse at Stage 1 end)
3. Step 6k: >8 tokens (recovery when WD changes)
4. **Step 20k: >50 tokens** (stays diverse through Stage 2)
5. **Step 30k: >50 tokens** (if extended test)

**If all true:** Fix solves the problem ✅

### Failure Indicators (Fix Doesn't Work)

**Any of these means fix needs more investigation:**
1. Step 5k: >5 tokens (not collapsing as expected)
2. Step 6k: <5 tokens (no recovery)
3. Step 10k: <5 tokens (recovery is temporary)
4. **Step 20k: <10 tokens** (collapses again in Stage 2)
5. Highly volatile pattern (tokens jump around unpredictably)

---

## What Each Test Output Looks Like

### Example output (CSV format):
```
Step,Loss,UniquePredictions,LR,WD
1,11.08,20,0.000004,0.10
1000,10.82,16,0.001432,0.10
2000,10.82,8,0.001324,0.10
...
5000,10.34,2,0.001000,0.10
6000,7.32,10,0.000800,0.00
...
20000,6.50,XX,0.000001,0.00
```

The critical column is **UniquePredictions** (token count)

---

## Commands to Report Results

After running, send me:

1. **Raw output files** (the .txt files)
2. **Summary table** showing steps 1k, 5k, 6k, 10k, 20k token counts for each model
3. **Final token counts at end of each test**

Example summary:
```
256-hidden 20k test:
Step 1k:   16 tokens
Step 5k:    2 tokens (Stage 1 end)
Step 6k:   10 tokens (WD change)
Step 10k:  12 tokens
Step 20k:  XX tokens (CRITICAL - what is this?)

512-hidden 20k test:
Step 20k:  XX tokens

768-hidden 30k test:
Step 30k:  XX tokens
```

---

## Troubleshooting

### If test fails to run:
- Check that `src/bitnet/` files are in place
- Check that WikiText dataset is available locally
- Check GPU is being used (should see significant speedup)

### If token counts look weird (too high/low):
- Check if the numbers make sense (1-50000 range is reasonable)
- Look at loss trajectory (should decrease over training)

### If tests complete but results are inconclusive:
- We might need to adjust test parameters
- Might need longer training periods
- Might need to log additional metrics

---

## Timeline on GPU

**Optimistic (all tests complete):**
- Phase 1: 30 seconds
- Phase 2: 2 minutes
- Phase 3: 3 minutes
- Phase 4: 5 minutes
- **Total: ~10 minutes**

**Realistic:** 15-20 minutes including any retries or issues

---

## What We'll Learn

**If fix works (>50 tokens at step 20k):**
- WD schedule is the issue, not model size or LR
- Can use 256-hidden BitNet with correct hyperparameters
- Problem is completely solved

**If fix doesn't work (<10 tokens at step 20k):**
- WD schedule alone isn't the solution
- Need to also adjust LR or use larger model
- Back to hypothesis: reduced LR needed for small models
- Next step: test with LR=0.0005 instead

---

## Files You Need to Copy

```
/Users/dan/bitnet/src/bitnet/
  ├── config.py
  ├── data.py
  ├── linear.py
  ├── quant.py
  ├── train.py (CRITICAL - has WD=0.0 fix)
  └── transformer.py

/Users/dan/bitnet/examples/
  ├── test_model_scaling.py (MAIN TEST SCRIPT)
  └── train_simple.py (reference)
```

---

## Summary

**Main Question:** Does WD=0.0 in Stage 2 fix the degenerate decoding problem?

**How to Answer:** Run test_model_scaling.py for 20k steps and check final token count

**Success Threshold:** >50 tokens at end of training (vs. current 2-6 tokens)

**Time Required on GPU:** ~10 minutes for all tests

**Expected Outcome:** Clear yes/no answer whether the fix works
