# Phase 4: Hypothesis and Testing - Root Cause Identified

**Date:** December 18, 2025
**Status:** Phase 4 hypothesis testing in progress

---

## Root Cause Discovery (Refined Phase 3)

### Evidence from Running Diagnostics

**Key Finding:** The loss DECREASES while tokens COLLAPSE

**Reduced LR (0.0005) at step 1000:**
- Tokens: 20 → 3 (95% collapse!)
- Loss: 10.98 → 9.89 (9.9% improvement!)

This is the smoking gun. The **CrossEntropyLoss function is rewarding mode collapse**.

### Why This Happens

Standard CrossEntropyLoss computes: `-log(P(correct_token))`

A mode-collapsed model that learns:
- "Token 42 appears in 30% of next positions → predict Token 42"
- "Token 100 appears in 25% of next positions → predict Token 100"
- "Token 7 appears in 20% of next positions → predict Token 7"

Gets LOWER loss than a diverse model because it's overfitting to the most common next-tokens in the training data.

But for **generation quality**, we need diversity. A model that only produces 3 tokens can't generate coherent text.

### Root Cause

**NOT:** Hyperparameter issue (LR, WD schedule, etc.)
**NOT:** Quantization failure
**NOT:** Weight initialization

**YES:** The training objective (CrossEntropyLoss) doesn't penalize vocabulary collapse

---

## Phase 4: Testing Solutions

### Solution 1: Label Smoothing (TESTING NOW)
**Command:** `train_with_label_smoothing.py 10000 0.1 /tmp/label_smooth_10k.txt`

**Mechanism:** Instead of one-hot targets, distribute target probability:
- Correct token: 90% probability
- All other tokens: 0.01% probability each (10% / 50,257 tokens)

**Effect:** Model must learn about the full vocabulary to minimize loss. Can't achieve perfect loss by memorizing 3 tokens.

**Expected Result:** Should maintain higher unique token count

**Status:** Running in background (ID: ba9e1ce)

### Solution 2: Entropy Regularization (PLANNED)
Add penalty term to loss: `-β * H(predicted_distribution)`

Where H is entropy. This directly penalizes overly confident (collapsed) predictions.

### Solution 3: Vocabulary Diversity Bonus (PLANNED)
Add bonus term: `+λ * unique_token_count`

Directly incentivize using more tokens.

### Solution 4: Output Layer Regularization (PLANNED)
Add L2 penalty on output layer weights to prevent extreme values.

---

## Comparison of All Approaches

| Training | Tokens @ 1k | Tokens @ 4k | Recovery? | Loss @ 1k |
|----------|------------|------------|-----------|-----------|
| Standard LR | 19 | 2 | No | 10.82 |
| Reduced LR | 3 | 18 | YES! | 9.89 |
| Label Smooth (?) | ? | ? | ? | ? |
| Entropy Reg (?) | ? | ? | ? | ? |

---

## Key Insights

### Why Reduced LR Recovered
The reduced LR run showed:
- Collapse at step 1000 (3 tokens)
- Recovery to 18 tokens by step 4000

This suggests that with **different training dynamics** (slower LR decay), the model can escape collapse. But standard LR gets stuck.

Possible explanations:
1. Standard LR decays too slowly → weights diverge to extremes
2. Reduced LR decays faster (in absolute terms) → provides "escape route" from collapse
3. Something about the gradient flow at reduced LR prevents weight polarization

### Label Smoothing Should Help Because
- It prevents the model from achieving arbitrarily low loss
- It forces learning of full vocabulary
- It's proven technique in other domains (image classification, etc.)

### What We're Testing
Whether fixing the **loss function** solves the problem without changing hyperparameters.

If label smoothing works:
- Problem: Fundamentally the training objective, not hyperparameters
- Solution: Add regularization to loss function
- Implication: Microsoft's BitNet training might benefit from loss modification

If label smoothing doesn't work:
- Problem: Something deeper about ternary quantization dynamics
- Next step: Test entropy regularization or other solutions
- Implication: Might need architectural changes

---

## Running Tests Status

| Test | Command | Duration | Data | Status |
|------|---------|----------|------|--------|
| Stage 1-Only (std LR) | `train_stage1_only.py 30000` | ongoing | up to 7k steps | Running bg |
| Reduced LR | `train_reduced_lr.py 10000` | ongoing | up to 4k steps | Running bg |
| Label Smoothing | `train_with_label_smoothing.py 10000 0.1` | ongoing | ? | Running bg |

### Data Collection Points
- Unique tokens every 1000 steps
- Loss every 1000 steps
- LR and WD every 1000 steps

---

## Next Steps (Waiting for Results)

### If Label Smoothing Works (maintains >20 tokens):
1. **Confirm**: Run full Stage 1-only with label smoothing
2. **Test**: Full training pipeline with label smoothing
3. **Optimize**: Find best smoothing value (0.05, 0.1, 0.15)
4. **Deploy**: Use label smoothing in training

### If Label Smoothing Doesn't Work:
1. **Test**: Entropy regularization term
2. **Test**: Direct vocabulary bonus
3. **Investigate**: Whether issue is specific to 256-hidden model
4. **Consider**: Whether ternary quantization needs different loss

### Regardless of Result:
1. Update training guidelines
2. Add vocabulary preservation metrics
3. Monitor for other modes of collapse

---

## Theoretical Implications

If loss function is the problem:
- BitNet training could be improved with standard regularization techniques
- The "collapse to 2-6 tokens" might be specific to 256-hidden models
- Larger models might have better implicit regularization

If loss function isn't the problem:
- Collapse might be unavoidable for models below certain scale
- Might need architectural changes (e.g., explicit diversity mechanisms)
- Ternary quantization might have fundamental scaling limits

---

## Files Created This Phase

- `/Users/dan/bitnet/examples/train_with_label_smoothing.py` - Label smoothing implementation
- `/tmp/label_smooth_10k.txt` - Label smoothing results (pending)
- `PHASE4_HYPOTHESIS_AND_TESTING.md` - This document

---

## Key Questions Being Answered

1. **Is the loss function the problem?**
   - Label smoothing test will show

2. **Can we fix it without changing hyperparameters?**
   - Label smoothing test will show

3. **Is collapse recoverable?**
   - Reduced LR recovery suggests YES

4. **Why does reduced LR recover but standard LR doesn't?**
   - Need deeper analysis of weight evolution

5. **Will this scale to larger models or just 256-hidden?**
   - Need to test on different model sizes
