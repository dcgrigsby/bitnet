# WikiText Training Data Integration Design

## Overview

Replace dummy data in `train_simple.py` with real training data from WikiText-2 (Hugging Face).

## Architecture

### New Files
- `src/bitnet/data.py` - WikiTextDataLoader class
- `tests/fixtures/dummy_data.py` - Move existing `create_dummy_dataloader` here

### WikiTextDataLoader (`src/bitnet/data.py`)

A simple dataloader that:
- Downloads WikiText-2 from Hugging Face (cached locally)
- Tokenizes and pads sequences to `seq_len`
- Yields batches via iteration
- Constructor takes: `tokenizer`, `batch_size`, `seq_len`, `num_steps` (optional, how many batches to generate)

### train_simple.py Changes

- Replace `create_dummy_dataloader()` call with `WikiTextDataLoader()`
- Remove dummy data generation logic
- Accept optional CLI argument for number of steps (default 48)
- Train for the specified number of steps instead of fixed epochs

### tests/fixtures/dummy_data.py

- Move `create_dummy_dataloader` from `src/bitnet/train.py` unchanged
- Tests can import it when needed

### justfile Update

```
train-simple steps='48':
    uv run python examples/train_simple.py {{steps}}
```

## Data Flow

1. `train_simple.py` creates `WikiTextDataLoader(tokenizer, batch_size=4, seq_len=32, num_steps=48)`
2. Dataloader yields batches of tokenized WikiText-2 sequences
3. Training loop runs for specified steps (vs epochs)
4. Same `train_epoch()` interface works unchanged

## Behavior

- Default: `just train-simple` runs for 48 steps (~1 minute)
- Custom: `just train-simple 100` runs for 100 steps
- WikiText-2 data is cached locally after first download
