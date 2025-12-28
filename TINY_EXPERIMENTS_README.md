# Tiny BitNet Validation Experiments

Quick validation experiments to test BitNet training infrastructure with small models and simple datasets.

## Purpose

The 95M parameter model trained for 400K steps on FineWeb-Edu is not producing coherent output. These experiments help determine if the issue is:

1. **Infrastructure broken** (quantization, gradients, optimizer) → These tiny experiments will fail
2. **Model/data mismatch** (too small for complex task) → These tiny experiments will succeed

## Experiments

### 1. Arithmetic Experiment (FASTEST - 30 minutes)

**Model:** 5M parameters
**Dataset:** Synthetic arithmetic (2 + 3 = 5)
**Vocabulary:** 32 tokens (digits + operators)
**Training time:** ~30 minutes on RTX 3060

**Purpose:** Quick sanity check that BitNet quantization and gradients work.

**How to run:**
```bash
./start_arithmetic_experiment.sh
```

**Expected outcome:**
- Loss drops from ~3.5 to ~0.5
- Model correctly completes arithmetic: "5 + 3 = 8"
- If this fails → Infrastructure has bugs
- If this works → Move to TinyStories

---

### 2. TinyStories Experiment (RECOMMENDED - 2 hours)

**Model:** 12M parameters
**Dataset:** TinyStories (simple children's stories)
**Vocabulary:** 2,048 tokens (custom trained)
**Training time:** ~2 hours total (10 min tokenizer + 1.5 hrs training)

**Purpose:** Validate BitNet works on natural language with appropriately-sized model.

**How to run:**
```bash
./start_tinystories_experiment.sh
```

**Expected outcome:**
- Loss drops from ~7 to ~1.5-2.0
- Model generates coherent simple stories
- Example: "Once upon a time there was a little boy named Tim. He liked to play with his ball."

**If this succeeds:**
- ✅ BitNet training infrastructure is working correctly
- → Problem with 95M model is likely vocabulary/data mismatch
- → Next steps: Try 95M with TinyStories OR train longer OR scale up

**If this fails:**
- ❌ BitNet implementation has bugs
- → Debug: quantization, STE gradients, data pipeline

---

## Files Created

### Configuration
- `src/bitnet/config_tiny.py` - Model configs for tiny experiments
  - `BitNetConfigTiny` - 12M params for TinyStories
  - `BitNetConfigArithmetic` - 5M params for arithmetic

### Data Loaders
- `src/bitnet/data_tinystories.py` - TinyStories streaming loader
- `src/bitnet/data_arithmetic.py` - Arithmetic data generator & tokenizer

### Training Scripts
- `train_tinystories_tokenizer.py` - Train custom 2K vocab tokenizer
- `train_bitnet_tiny.py` - Train 12M model on TinyStories
- `train_bitnet_arithmetic.py` - Train 5M model on arithmetic

### Launch Scripts
- `start_arithmetic_experiment.sh` - Quick arithmetic validation
- `start_tinystories_experiment.sh` - Full TinyStories experiment

## Recommended Workflow

### Step 1: Quick Validation (30 min)
```bash
./start_arithmetic_experiment.sh
```

Check if model learns basic arithmetic. If yes, move to Step 2.

### Step 2: Natural Language (2 hrs)
```bash
./start_tinystories_experiment.sh
```

Check if model produces coherent stories. If yes, infrastructure is good!

### Step 3: Debug or Scale

**If both experiments succeed:**
- Infrastructure is working ✅
- Try 95M model with TinyStories dataset
- Or train 95M for 10× longer (4M steps)
- Or scale to 200M+ parameters for 32K vocabulary

**If experiments fail:**
- Debug BitNet quantization
- Check STE gradient flow
- Verify data pipeline
- Inspect weight distributions

## Expected Results

### Arithmetic (Success)
```
Step 9500: Testing arithmetic...
  5 + 3 = 8
  12 - 7 = 5
  8 * 6 = 48
  20 / 4 = 5
  15 + 25 = 40
```

### TinyStories (Success)
```
Step 29000: Generating samples...
  [Once upon a time] → Once upon a time there was a little girl named Lily.
                        She loved to play outside. One day she saw a big tree
                        and wanted to climb it.
```

### Failure Indicators
- Loss stuck above 5 (not learning)
- Repetitive output: "the the the the"
- Gibberish: "xkq zzt mlp"
- NaN/Inf in loss

## Dependencies

```bash
pip install torch transformers datasets tokenizers tqdm numpy
```

## Storage Requirements

- Arithmetic: ~100 MB (checkpoints + logs)
- TinyStories: ~500 MB (tokenizer + checkpoints + logs)

## References

- [TinyStories Paper](https://arxiv.org/abs/2305.07759) - Microsoft Research
- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [TinyGSM Arithmetic](https://arxiv.org/abs/2312.09241)

---

**Status:** Ready to run
**Last Updated:** 2025-12-26
