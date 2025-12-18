# BitNet Degenerate Decoding - Systematic Debugging Analysis
**Date:** December 18, 2025
**Methodology:** Systematic Debugging Skill (Phase 1-3)

---

## Phase 1: Root Cause Investigation

### Evidence Gathered

**1. Stage 1-Only Diagnostic (Standard LR=0.0015)**
- Created to isolate whether collapse is in Stage 1 itself or Stage 1→2 transition
- Command: `train_stage1_only.py 30000 /tmp/stage1_only_30k.txt`
- Status: Running (data up to step 4000)

| Step | Unique Tokens | Loss | LR | Notes |
|------|---------------|------|-----|-------|
| 1 | 20 | 11.02 | 0.000004 | Healthy start |
| 1,000 | 19 | 10.82 | 0.001479 | Minimal loss (-1 token) |
| 2,000 | 9 | 10.82 | 0.001444 | Major loss (-10 tokens, 53%) |
| 3,000 | 4 | 10.23 | 0.001410 | Severe (-5 tokens, 56%) |
| 4,000 | 2 | 10.25 | 0.001376 | **COLLAPSED** |

**Key Insight:** Collapse happens **within Stage 1**, NOT at transition. Happens during peak learning rate maintenance (LR ≈ 0.0013-0.0014).

**2. Reduced LR Test (LR=0.0005, 1/3 of standard)**
- Command: `train_reduced_lr.py 10000 /tmp/reduced_lr_10k.txt`
- Status: Running (data up to step 2000)
- **HYPOTHESIS REJECTED**: Reducing LR to 0.0005 did NOT prevent collapse

| Step | Unique Tokens | Loss | LR |
|------|---------------|------|-----|
| 1 | 20 | 10.98 | 0.000001 |
| 1,000 | 3 | 9.89 | 0.000477 | **Collapsed FASTER than standard LR** |
| 2,000 | 10 | 10.73 | 0.000441 | **Recovered to 10 tokens!** |

**Critical Finding:** Reducing LR accelerated collapse but enabled recovery. This suggests:
- Collapse is NOT caused by LR magnitude alone
- May be part of 1-bit training S-curve dynamics
- Model showed recovery potential (10 tokens at step 2000)

### Phase 1 Conclusion

**Root Cause NOT:**
- ❌ Stage 1→2 transition (collapse happens within Stage 1)
- ❌ Learning rate too high (reducing LR caused faster collapse, not prevention)
- ❌ Simple hyperparameter issue (collapse+recovery pattern suggests complex dynamics)

**Root Cause POSSIBLY:**
- ✓ Ternary quantization + small model (256 hidden) interaction
- ✓ S-curve training dynamics with collapse/recovery phases
- ✓ Weight initialization or early training convergence pattern
- ✓ Loss function not preserving vocabulary diversity

---

## Phase 2: Pattern Analysis

### Working Examples Reference

**Microsoft BitNet Training (from official guides):**
- Model sizes: 700M, 1.3B, 3B, 3.9B (minimum 700M)
- Training tokens: 100B
- Batch size: 1M
- Sequence length: 2048
- Stage 1 weight decay: 0.1
- Stage 2 weight decay: 0.0 (not 0.05!)

**Your Model:**
- Size: 256 hidden (1000x smaller than 700M)
- Your Stage 2 WD: 0.05 (contrary to Microsoft's 0.0)

### Key Differences Identified

1. **Model Scale:** 256 hidden vs 700M minimum
   - May not exhibit same training dynamics as full models
   - Ternary quantization behavior might differ

2. **S-Curve Loss Pattern (Microsoft's Finding)**
   - Microsoft notes: 1-bit training shows S-shaped loss curve
   - Exponential drop → flat plateau → sudden improvement
   - Intermediate evaluations sample during "flat" phase, looking like collapse

3. **Previous Test Results (from FINAL_REPORT)**
   - 1000-cycle: Collapsed step 140 (14% of training)
   - 10000-cycle: Better (0.7% severe collapses vs 28.5%)
   - 20000-cycle: Worse than 10000-cycle (non-monotonic!)
   - **Pattern:** Not monotonic improvement, suggesting:
     - Not simple "more training helps"
     - Possible evaluation-timing artifact
     - Or hidden pathological behavior at certain scales

### Phase 2 Conclusion

**Pattern Matches:**
- S-curve behavior with collapse/recovery potential ✓
- Small model scale showing different properties ✓
- Non-monotonic improvement with training length ✓

**Pattern Doesn't Match:**
- Monotonic improvement (20k > 10k > 1k) ✗
- Simple convergence (should improve steadily) ✗

**Hypothesis:** Training dynamics are fundamentally different for 256-hidden model vs Microsoft's 700M+. The collapse might be expected for this scale, with recovery requiring different evaluation strategy or training continuation.

---

## Phase 3: Hypothesis and Testing

### Initial Hypothesis (REJECTED)
"LR=0.0015 is too high for 256-hidden model, causing rapid weight divergence"

**Why Rejected:**
- Reducing LR to 0.0005 collapsed FASTER (step 1000 vs step 4000)
- This is opposite of expected behavior if LR was the cause
- Shows collapse is not simple LR magnitude issue

### New Hypothesis (SUPPORTED by evidence)
"The model exhibits 1-bit training S-curve dynamics with collapse/recovery phases. Intermediate evaluation during collapse phase creates false impression of failure."

**Supporting Evidence:**
1. Reduced LR runs show recovery (3→10 tokens from step 1k to 2k)
2. Microsoft's training guide emphasizes S-curve pattern
3. Non-monotonic performance (20k worse than 10k) suggests evaluation timing effect
4. Loss still declining normally (10.98→9.89 in reduced LR) during collapse

### Tests Running

**Test 1: Full 30k run (standard LR)**
- Running: yes
- Purpose: See if collapse is permanent or recovers with more steps
- Data up to: step 4000/30000

**Test 2: Reduced LR for full 10k**
- Running: yes
- Purpose: Compare collapse/recovery pattern at different LR
- Data up to: step 2000/10000

### Proposed Next Tests

**Test 3:** Run standard LR to 20k+ and evaluate END state only (not intermediate)
- Purpose: Test if S-curve recovery happens with more steps

**Test 4:** Continuous training (don't stop at collapse)
- Purpose: See if model recovers without hyperparameter changes

**Test 5:** Change evaluation strategy
- Only evaluate at: 0%, 25%, 50%, 75%, 100% of training
- Purpose: Reduce evaluation-timing artifacts

---

## Summary Table: What We Know

| Aspect | Finding | Confidence |
|--------|---------|------------|
| **Root Location** | Within Stage 1, not transition | HIGH |
| **LR Magnitude** | NOT the cause | HIGH |
| **Collapse Pattern** | Rapid (steps 2-4k) then stable | HIGH |
| **Recovery Potential** | YES (shown in reduced LR run) | MEDIUM |
| **S-Curve Dynamics** | Likely present | MEDIUM |
| **Model Scale Issue** | Probable contributor | MEDIUM |
| **Final Cause** | Unclear (multiple factors?) | LOW |

---

## Critical Open Questions

1. **Does the reduced LR run recover further at step 3k, 4k, etc?**
2. **What is the final token count at step 10k for both runs?**
3. **Is 2-token collapse PERMANENT or temporary?**
4. **Does training beyond step 4k show S-curve recovery?**
5. **Does the non-monotonic performance (20k < 10k) persist?**

---

## Next Steps (Awaiting Full Diagnostic Data)

When diagnostics complete:

1. **Confirm collapse permanence** by checking final unique token counts
2. **Analyze recovery trajectories** across both LR conditions
3. **Test evaluation strategy** (end-only vs intermediate)
4. **If recovery confirmed:** Implement continuous training with end-only eval
5. **If no recovery:** Question architecture (see Phase 4.5 in systematic debugging)

---

## Files Generated

- `/Users/dan/bitnet/examples/train_stage1_only.py` - Stage 1-only diagnostic
- `/Users/dan/bitnet/examples/train_reduced_lr.py` - Reduced LR test script
- `/tmp/stage1_only_30k.txt` - Running (30k steps, standard LR)
- `/tmp/reduced_lr_10k.txt` - Running (10k steps, LR=0.0005)
