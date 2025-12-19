# Scaling Test Predictions and Analysis

**Test Start Time:** December 19, 2025
**Estimated Completion:** ~10-15 minutes
**Models Testing:** 256, 512, 768, 1024-hidden in parallel

---

## What the Test Will Show Us

### Setup Recap
All models train with:
- **Standard LR:** 0.0015 (per Microsoft BitNet papers)
- **Corrected WD:** Stage 1=0.1, Stage 2=0.0 (per papers, NOT 0.05)
- **Training:** 10,000 steps Stage 1 only
- **Metric:** Unique next-token predictions (collapse if <10)

### Key Question Being Answered
**"Does the degenerate decoding problem disappear as we increase model size?"**

---

## Three Possible Outcomes

### Outcome A: Scaling Fixes It ✅ (MOST LIKELY BASED ON THEORY)

**Expected pattern:**
```
Model Size | Step 1k | Step 4k | Step 10k | Interpretation
-----------|---------|---------|----------|-------------------
256        |   15    |    2    |    2     | Still collapses
512        |   40    |   20    |   30     | Partial collapse, recovery
768        |   80    |   60    |  100     | Marginal stability
1024       |  200+   |  300+   |  500+    | Stable, healthy
```

**Why this is most likely:**
1. Reduced LR (0.0005) enabled recovery on 256-hidden
2. Means capacity exists, just needs right optimization
3. Larger models have more capacity → better escape from collapse
4. BitNet papers (700M+) don't show collapse
5. Scaling laws favor this outcome

**If true, conclusion:**
- BitNet has minimum viable model size (~512-768 hidden)
- Microsoft's LR (0.0015) is correct for their scale
- Need models of sufficient capacity for stable training
- This aligns with principle: bigger models ≠ same behavior

---

### Outcome B: No Clear Threshold ❌ (LESS LIKELY)

**Expected pattern:**
```
Model Size | Step 1k | Step 4k | Step 10k | Interpretation
-----------|---------|---------|----------|-------------------
256        |   10    |    2    |    2     | Collapses
512        |   12    |    2    |    2     | Still collapses
768        |   15    |    2    |    2     | Still collapses
1024       |   18    |    2    |    2     | Even 1024 collapses!
```

**Why this might happen:**
1. LR (0.0015) is fundamentally too aggressive for ternary quantization
2. Might apply even to larger models if truly small
3. Would contradict Microsoft's papers (which say it works)

**If true, conclusion:**
- LR needs to be scaled with model size
- Even 1024-hidden not large enough
- Need to create adaptive LR scaling function
- Might need LR = 0.0015 * sqrt(256 / hidden_size) or similar

---

### Outcome C: Mixed/Surprising Pattern ⚡ (POSSIBLE)

**Possible patterns:**
```
Example 1: Exponential improvement
256 (2 tokens) → 512 (5) → 768 (50) → 1024 (500)
→ Clear threshold around 512-768

Example 2: Plateau effect
256 (2) → 512 (3) → 768 (4) → 1024 (5)
→ Size doesn't matter much, need LR fix

Example 3: Non-monotonic
256 (2) → 512 (100) → 768 (10) → 1024 (200)
→ Some unexpected interaction we need to investigate
```

**If true, conclusion:**
- Requires deeper investigation
- Might discover new insight about ternary quantization
- Could be complex interaction of multiple factors

---

## Why Scaling Would Make Sense Theoretically

### 1. Optimization Landscape Complexity
- Larger models have more parameters → more dimensions
- More dimensions → more possible "escape routes" from collapse attractor
- Smaller models have fewer routes → easier to get trapped

### 2. Ternary Quantization Sensitivity
- Ternary quantization ({-1, 0, 1}) is a discrete operation
- Discretization effects matter more with fewer weights
- Larger models might naturally "smooth out" quantization effects

### 3. Gradient Flow
- Larger networks: Gradients flow through more paths
- More paths → better gradient distribution
- Better distribution → less likely to converge to degenerate solution

### 4. Loss Landscape Properties
- Larger models: Loss landscape is more "rugged"
- More minima → more ways to achieve good loss with diversity
- Smaller models: Fewer minima → collapse might be the primary minimum

---

## What Will Happen During the Test

### Timeline (Approximate)

**Minutes 1-2:** All models train through warmup phase
- All should maintain ~20 tokens (random predictions still)
- Loss starts declining as model learns

**Minutes 3-5:** Peak learning rate phase (step ~1000-3000)
- This is where collapse typically happens
- Prediction: 256 collapses here, others less so
- Should see divergence between model sizes

**Minutes 6-10:** Main training phase (step ~3000-10000)
- 256 likely stays collapsed (2-6 tokens)
- Larger models should show whether they recover or not
- Should see clear pattern emerging

---

## How to Analyze Results

### Step 1: Quick Comparison Table
```bash
echo "Size | Step 1k | Step 4k | Step 10k"
for size in 256 512 768 1024; do
  s1k=$(grep "^1000," /tmp/scale_${size}_10k.txt | cut -d, -f3)
  s4k=$(grep "^4000," /tmp/scale_${size}_10k.txt | cut -d, -f3)
  s10k=$(grep "^10000," /tmp/scale_${size}_10k.txt | cut -d, -f3)
  echo "$size  |  $s1k  |  $s4k  | $s10k"
done
```

### Step 2: Plot Trajectory
Look for these patterns:
- **Collapse pattern:** Rapid 20→2 drop (like 256-baseline)
- **Recovery pattern:** Collapse then recovery (like reduced LR test)
- **Stable pattern:** Stays above 50 throughout

### Step 3: Identify Threshold
What's the minimum hidden_size where tokens stay >100?

### Step 4: Compare to Theory
Do results match one of our three outcomes?

---

## Decision Tree After Results

```
Does any model stay diverse (>100 tokens)?
│
├─ YES, at 512-hidden
│  └─ Conclusion: Scaling fixes it
│     Action: Use 512+ hidden models
│
├─ YES, at 768-hidden
│  └─ Conclusion: Clear scaling threshold
│     Action: Document 768 as minimum
│
├─ YES, at 1024+ hidden
│  └─ Conclusion: Need much larger models
│     Action: Combine with reduced LR or scale LR
│
└─ NO, all collapse
   └─ Conclusion: LR needs adjustment
      Action: Create LR scaling function
         or use reduced LR for all small models
```

---

## Expected Insights

### Most Valuable Finding
If larger models work, it validates:
1. BitNet is sound technology
2. Papers' recommendations are correct
3. Our 256-hidden model is just below minimum
4. Easy fix: use larger models for this approach

### Second Most Valuable Finding
If no scaling helps but reduced LR does:
1. Proves optimization dynamics are the issue
2. LR must be scaled with model size
3. Opens possibility of BitNet at any scale
4. More work but more flexible

### Least Likely But Interesting
If scaling partially helps:
1. Suggests complex interaction
2. Might discover new principle
3. Could be valuable insight for future work

---

## Why This Matters

This test **completes the investigation systematically**:

1. **Phase 1:** Found collapse location (Stage 1)
2. **Phase 2:** Found optimization dynamics issue
3. **Phase 3:** Found reduced LR enables recovery
4. **Phase 4:** Found loss/gradient fixes don't work
5. **Phase 5:** Determining if it's model size or LR scaling

After this test, we'll have **clear evidence** for which direction to go:
- Use larger models
- Scale down LR
- Or both with clear scaling law

This is much better than guessing or trying random changes.

---

## Takeaway

**The real question we're answering is:**

"Is BitNet training fundamentally incompatible with 256-hidden models,
or can we make it work by adjusting one hyperparameter (LR)?"

The answer will guide whether 1-bit training can be practical at smaller scales,
which has implications for deployment, efficiency, and research directions.

**This test will give us that answer with high confidence.**
