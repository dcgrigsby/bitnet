# GPU Testing Package - File Index

**All files needed to verify the BitNet WD fix are ready**

---

## START HERE

Read in this order:

1. **README_GPU_VERIFICATION.md** (5 min read)
   - Quick overview
   - What the problem is
   - How long it takes

2. **GPU_VERIFICATION_PLAN.md** (10 min read)
   - Exact commands to run
   - What to expect at each step
   - How to interpret results

3. **Run the tests** (5-10 min on GPU)
   - Follow the commands in the plan

4. **Report back results**
   - Token counts at step 20k
   - Any issues encountered

---

## Documentation Files

### For GPU Verification (Required)
- **README_GPU_VERIFICATION.md** - Quick start guide and checklist
- **GPU_VERIFICATION_PLAN.md** - Exact test procedure with expected outputs
- **INVESTIGATION_SUMMARY_FOR_GPU_TESTING.md** - Full background and context

### For Understanding the Investigation (Optional)
- **SYSTEMATIC_DEBUGGING_FINAL_REPORT.md** - Complete Phase 1-4 analysis
- **CRITICAL_FINDING_WD_FIX.md** - Discovery of WD=0.0 fix
- **SCALING_TEST_FINAL_RESULTS.md** - Results from all 4 model sizes
- **INVESTIGATION_TIMELINE.md** - Full journey through debugging phases

### Original Problem Evidence
- **FINAL_REPORT.txt** - Shows the original degenerate decoding problem

---

## Testing Scripts

### Main Test Script (USE THIS)
**File:** `examples/test_model_scaling.py`

**What it does:** Trains BitNet models of different sizes and logs unique token counts

**Usage:**
```bash
python examples/test_model_scaling.py <steps> <hidden_size> <output_file>
```

**Example:**
```bash
python examples/test_model_scaling.py 20000 256 /tmp/gpu_verify_256_20k.txt
```

**Output:** CSV file with columns:
- Step
- Loss
- UniquePredictions (TOKEN COUNT - this is what we care about)
- LR
- WD

### Diagnostic Scripts (Already Tested on CPU)
These were used to understand the problem, not needed for GPU verification:
- `examples/train_stage1_only.py` - Stage 1 isolation test
- `examples/train_reduced_lr.py` - Reduced LR testing
- `examples/train_with_label_smoothing.py` - Loss function testing
- `examples/train_aggressive_grad_clip.py` - Gradient testing

### Reference Script
- `examples/train_simple.py` - Basic training reference

---

## Source Files (Copy to GPU System)

### Critical (Has the WD fix)
- **src/bitnet/train.py** - Contains `TwoStageWDScheduler` with `stage2_wd = 0.0` (THE FIX)

### Required
- src/bitnet/config.py - Configuration
- src/bitnet/transformer.py - Model architecture
- src/bitnet/linear.py - BitLinear layer
- src/bitnet/quant.py - Quantization functions
- src/bitnet/data.py - Data loading
- src/bitnet/attention.py - Attention implementation
- src/bitnet/feedforward.py - Feedforward networks
- src/bitnet/__init__.py - Package init

---

## Quick Command Reference

### Copy to GPU system:
```bash
# All source files
cp -r src/bitnet/ /destination/

# Test script
cp examples/test_model_scaling.py /destination/examples/
```

### Run verification:
```bash
# Quick sanity check (5k steps)
python examples/test_model_scaling.py 5000 256 /tmp/quick_check.txt

# Critical test (20k steps, 256-hidden)
python examples/test_model_scaling.py 20000 256 /tmp/gpu_verify_256_20k.txt

# Confirmation test (20k steps, 512-hidden)
python examples/test_model_scaling.py 20000 512 /tmp/gpu_verify_512_20k.txt
```

### Check results:
```bash
# View last 3 lines (final step)
tail -3 /tmp/gpu_verify_256_20k.txt
tail -3 /tmp/gpu_verify_512_20k.txt
```

---

## What Each File Does

### Documentation

| File | Purpose | Reading Time |
|------|---------|--------------|
| README_GPU_VERIFICATION.md | Overview and checklist | 5 min |
| GPU_VERIFICATION_PLAN.md | Exact test procedure | 10 min |
| INVESTIGATION_SUMMARY_FOR_GPU_TESTING.md | Full background | 15 min |
| SYSTEMATIC_DEBUGGING_FINAL_REPORT.md | Detailed analysis | 20 min |
| CRITICAL_FINDING_WD_FIX.md | How we found the fix | 10 min |
| SCALING_TEST_FINAL_RESULTS.md | CPU test results | 10 min |

### Scripts

| File | Purpose | Need for GPU? |
|------|---------|--------------|
| test_model_scaling.py | Main verification test | **YES** |
| train_stage1_only.py | Stage 1 isolation (already tested) | No |
| train_reduced_lr.py | LR testing (already tested) | No |
| train_with_label_smoothing.py | Loss testing (already tested) | No |
| train_aggressive_grad_clip.py | Gradient testing (already tested) | No |

### Source Code

| File | Purpose | Has Fix? |
|------|---------|----------|
| train.py | Training and schedulers | **YES** |
| config.py | Model configuration | No |
| transformer.py | Model architecture | No |
| linear.py | BitLinear layer | No |
| quant.py | Quantization | No |
| data.py | Data loading | No |

---

## The Fix (For Reference)

**Location:** `src/bitnet/train.py`, line 93

**Before (WRONG):**
```python
self.stage2_wd: float = 0.05
```

**After (CORRECT):**
```python
self.stage2_wd: float = 0.0
```

This matches Microsoft's official BitNet training specification.

---

## Expected Results

### Success (Fix Works)
```
256-hidden, step 20k: 50+ tokens ✅
```

### Failure (Fix Doesn't Work)
```
256-hidden, step 20k: <10 tokens ❌
```

---

## File Organization for GPU System

Suggested structure for copying:
```
/destination/
├── src/
│   └── bitnet/
│       ├── __init__.py
│       ├── attention.py
│       ├── config.py
│       ├── data.py
│       ├── feedforward.py
│       ├── linear.py
│       ├── quant.py
│       ├── train.py          ← HAS THE FIX
│       └── transformer.py
├── examples/
│   └── test_model_scaling.py ← MAIN TEST SCRIPT
└── docs/
    ├── README_GPU_VERIFICATION.md
    ├── GPU_VERIFICATION_PLAN.md
    └── INVESTIGATION_SUMMARY_FOR_GPU_TESTING.md
```

---

## Verification Checklist

Before running tests:
- [ ] Have GPU system ready
- [ ] Have copied all files from above
- [ ] Have read README_GPU_VERIFICATION.md
- [ ] Have read GPU_VERIFICATION_PLAN.md

While running tests:
- [ ] Quick 5k test passes
- [ ] 20k test for 256-hidden completes
- [ ] 20k test for 512-hidden completes
- [ ] Outputs look reasonable (loss 6-11, tokens 1-50)

After running tests:
- [ ] Note final token counts at step 20k
- [ ] Note any errors or issues
- [ ] Ready to report results

---

## Results Template

When you run the tests and get results, report:

```
GPU System Used: [e.g., RTX 3060, V100, etc]
Test Duration: [actual time taken]

256-hidden, 20000 steps:
  Final token count: XX
  Status: [Pass/Fail]

512-hidden, 20000 steps:
  Final token count: XX
  Status: [Pass/Fail]

Overall Assessment:
  Fix works: [Yes/No]
  Confidence: [High/Medium/Low]

Any issues encountered:
  [list any problems]
```

---

## Getting Help

If you get stuck:

1. **Check README_GPU_VERIFICATION.md** section "Common Issues"
2. **Verify CUDA is working:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. **Check training is using GPU:**
   ```bash
   nvidia-smi (while test is running)
   ```

---

## Summary

**You have everything needed to verify the BitNet WD fix on GPU.**

Files needed:
- ✅ 3 documentation files (README + Plan + Summary)
- ✅ 1 test script (test_model_scaling.py)
- ✅ 8 source files (src/bitnet/)

Time to run:
- ✅ ~10 minutes total on GPU

Expected outcome:
- ✅ Clear yes/no on whether fix works
- ✅ Final token counts at step 20k

**Next step: Go to GPU system, follow GPU_VERIFICATION_PLAN.md, report results.**
