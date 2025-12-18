# Weight Quantization and Training Configuration Analysis
## Reproducing Microsoft BitNet b1.58 Results

**Date:** December 18, 2025
**Goal:** Identify the critical divergence between your implementation and Microsoft's official approach

---

## 1. QUANTIZATION CONSTRAINT ISSUE - NOT A PROBLEM

### Your Finding
- "981,223 weights outside [-1, 1] range despite ternary quantization"
- Marked as PROBLEM 4 in FINAL_REPORT.txt

### Microsoft's Documentation
From *The Era of 1-bit LLMs: Training Tips, Code and FAQ* (official training guide, page 5):

```python
def weight_quant(w):
    """ Per−tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
        w: a weight tensor with shape [d, k]
    Returns:
        u: a quantized weight with shape [d, k]
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e−5)
    u = (w * scale).round().clamp_(−1, 1) / scale
    return u
```

### The Critical Insight
**This is the dequantization step.** The process:
1. **Quantize** weights to {-1, 0, 1} via: `(w * scale).round().clamp(-1, 1)`
2. **Dequantize** by dividing by scale: `/ scale`

The **quantized** weights are always {-1, 0, 1}. The **dequantized** weights will be scaled versions like {-0.15, 0, 0.15} or {-2.3, 0, 2.3} depending on the scale factor.

**VERDICT:** Your observation of weights outside [-1, 1] is CORRECT and EXPECTED. This is not a bug.

---

## 2. WEIGHT DECAY SCHEDULE - CRITICAL DIVERGENCE FOUND

### Your Current Settings (from FINAL_REPORT and recent experiments)
```
Stage 1: weight_decay = 0.1
Stage 2: weight_decay = 0.05  ← ADJUSTED FROM 0.0
```

### Microsoft's Official Settings
From *The Era of 1-bit LLMs: Training Tips, Code and FAQ* (Table 2, all models):
```
Stage 1: weight_decay = 0.1
Stage 2: weight_decay = 0.0   ← NOT 0.05!
```

Also from page 3:
> "In the second stage, weight decay was disabled."

### Why This Matters for Your Problem
Your FINAL_REPORT states:
- PROBLEM 3: "Stage 2 Degradation - weight decay increase (0.0→0.05) insufficient"
- You then tested with 0.05, but the model still collapsed

**THIS WAS THE WRONG DIRECTION.** You increased weight decay when Microsoft disabled it entirely for Stage 2.

The theoretical reason (from page 3):
> "In mixed-precision training, the magnitude of the latent weights can be interpreted as a confidence score for the corresponding 1-bit weights. Consequently, a large weight decay leads to lower confidence scores for the 1-bit weights, causing them to change more frequently. To mitigate this, we removed weight decay for the second half of the training, allowing the model to converge rather than update frequently."

**ACTION ITEM:** Test with weight_decay = 0.0 in Stage 2, not 0.05.

---

## 3. S-SHAPED LOSS CURVE - YOUR EVALUATION STRATEGY

### Microsoft's Finding
From *The Era of 1-bit LLMs: Training Tips, Code and FAQ* (page 1):

> "The loss curve during high-precision training typically exhibits an exponential shape. However, the curve observed during 1-bit training often follows an S−shaped pattern. As illustrated in Figure 1a, there is a significant reduction in loss towards the end of the training process. A similar phenomenon has been observed in the training of binarized neural machine translation. **This suggests that intermediate evaluations may not accurately predict the final performance of 1-bit LLM training.**"

### What This Means for Your Results
Your 1000-cycle, 10000-cycle, and 20000-cycle runs show:
- Early collapse (step 140 in 1000-cycle)
- Recovery during mid-training
- Eventual collapse by end

**This could be the "flat" part of the S-curve, NOT actual model degeneration.**

The figure shows that for a properly trained model, you'd expect:
1. Rapid initial loss reduction (curved part of S)
2. Flat middle section (where model seems to plateau)
3. Sudden reduction at end (final steep part of S)

Your evaluations every 5 steps (1000-cycle) or every 20 steps (10000-cycle) might be sampling during the flat middle section, creating the false impression of collapse.

**IMPLICATION:** The non-monotonic behavior (10k better than 20k) might be an artifact of WHEN you evaluate, not the actual model state.

---

## 4. STAGE 2 TRANSITION - WHAT SHOULD HAPPEN

### Microsoft's Learning Rate Schedule
From *The Era of 1-bit LLMs: Training Tips, Code and FAQ* (page 1-2):

> "Halfway through the training process, we decayed the learning rate, a strategy that yielded better performance in our experiments. As illustrated in Figure 1b, we observed a **significant reduction in loss occurring when the learning rate was decayed, rather than at the end of the training phase.** This approach made the performance more predictable during the training process."

### What Your Report Shows
From FINAL_REPORT Section 5:
```
Stage 2 Average Unique: 3.6 tokens (MUCH WORSE than Stage 1!)
```

**This suggests the learning rate decay happened too aggressively or weight decay wasn't properly removed.**

According to Microsoft: "This approach made the performance more predictable during the training process." If Stage 2 is worse, this indicates the settings diverge from Microsoft's approach.

---

## 5. HYPERPARAMETER COMPARISON TABLE

| Parameter | Microsoft Official | Your Current | Your Previous |
|-----------|-------------------|--------------|---------------|
| Learning Rate Schedule | 2-stage with linear decay | Should match | TBD |
| Weight Decay Stage 1 | 0.1 | 0.1 ✓ | 0.1 ✓ |
| Weight Decay Stage 2 | **0.0** | 0.05 ✗ | 0.0 ✓ |
| Model Size | 700M, 1.3B, 3B, 3.9B | 256 hidden ✗ | N/A |
| Training Tokens | 100B | ? | ? |
| Batch Size | 1M | ? | ? |
| Sequence Length | 2048 | ? | ? |

---

## 6. ROOT CAUSE ANALYSIS - HYPOTHESIS

Your degenerate decoding problem likely stems from:

### Primary Suspect
**Incorrect weight decay schedule in Stage 2**
- You increased from 0.0 → 0.05 (opposite of Microsoft)
- This causes latent weights to have lower confidence scores
- Results in frequent weight flipping and instability

### Secondary Suspects
1. **Evaluation timing during flat S-curve section** - makes model appear worse than it is
2. **Non-monotonic training behavior** - 10k > 20k may be evaluation artifact
3. **Model size (256 hidden)** - Much smaller than Microsoft's 700M+; may not follow same scaling laws

### Not the Problem
- Quantization is working correctly
- Weights outside [-1, 1] is expected for dequantized values
- Loss function is valid

---

## 7. NEXT STEPS

### Phase 1: Fix Training Configuration (HIGHEST PRIORITY)
1. Change Stage 2 weight decay from 0.05 to 0.0
2. Verify learning rate schedule matches Microsoft's two-stage polynomial decay
3. Train for full duration WITHOUT intermediate evaluation
4. Only evaluate at end or at specific stage boundaries
5. Compare final loss to Microsoft's baseline

### Phase 2: Understanding S-Curve Behavior
1. Save model checkpoints only at:
   - Stage 1 midpoint (25%)
   - Stage 1 end (50%)
   - Stage 2 midpoint (75%)
   - Training end (100%)
2. Evaluate final model only, not intermediate
3. Track loss trajectory: should show S-shape with final reduction

### Phase 3: Model Size Consideration
Your 256 hidden model is 1000x smaller than Microsoft's smallest (700M).
- Verify it's designed for this scale
- May need architecture adjustments
- May not exhibit same collapse/recovery patterns

---

## Sources

- Microsoft BitNet Original Paper: [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- Microsoft Training Guide: [The Era of 1-bit LLMs: Training Tips, Code and FAQ](https://arxiv.org/abs/2402.17764)
- Quantization Overview: [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
