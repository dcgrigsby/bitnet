# Training Script Implementation Plan

## Overview
Create a clean, production-ready training script for the 95M BitNet model with full instrumentation and status monitoring.

---

## Architecture

```
bitnet/
├── src/bitnet/
│   ├── data.py              # Existing WikiText loader
│   ├── data_fineweb.py      # NEW: FineWeb-Edu streaming loader
│   ├── instrumentation.py   # NEW: Logging, checkpoints, provenance
│   └── generation.py        # NEW: Inference utilities (greedy, typical, stress)
└── train_bitnet_95m.py      # NEW: Main training script
└── check_training_status.py # NEW: Status monitoring script
```

---

## Module Breakdown

### 1. `src/bitnet/data_fineweb.py`
**Purpose:** Streaming dataloader for FineWeb-Edu

**Key Features:**
- Streaming mode (no full dataset download)
- Yields batches of shape [batch_size, seq_len]
- Supports num_steps iteration
- Proper tokenization with LLaMA-2 tokenizer
- Dataset fingerprinting for provenance

**Class:**
```python
class FineWebEduDataLoader:
    def __init__(self, tokenizer, batch_size, seq_len, num_steps):
        # Load streaming dataset
        # Initialize buffer for batch assembly

    def __iter__(self):
        # Yield batches until num_steps reached

    def get_fingerprint(self):
        # Return first 1000 token IDs for verification
```

---

### 2. `src/bitnet/instrumentation.py`
**Purpose:** All logging, checkpointing, and monitoring functionality

**Functions:**

```python
# Provenance & Setup
def create_run_directory(run_id: str) -> Path
def write_provenance(run_id: str, config: BitNetConfig)
def write_config(run_id: str, config: BitNetConfig, training_args: dict)
def write_dataset_fingerprint(run_id: str, dataloader: FineWebEduDataLoader)

# Checkpointing
def save_checkpoint(
    run_id: str,
    step: int,
    model: BitNetModel,
    optimizer: Optimizer,
    lr_scheduler: TwoStageLRScheduler,
    wd_scheduler: TwoStageWDScheduler,
    loss: float,
    mandatory: bool = False
)

def load_checkpoint(checkpoint_path: str) -> dict

# Metrics Logging
def log_scalars(
    run_id: str,
    step: int,
    loss: float,
    lr: float,
    wd: float,
    grad_norm: float,
    step_time_ms: float,
    **kwargs
)

def log_bitnet_metrics(
    run_id: str,
    step: int,
    model: BitNetModel
)

# Evaluation
def run_evaluation(
    run_id: str,
    step: int,
    model: BitNetModel,
    eval_dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> float

# Anomaly Detection
def detect_anomaly(
    loss: float,
    loss_history: list[float],
    entropy: float,
    max_repetition: int
) -> tuple[bool, str | None]

def trigger_anomaly_dump(
    run_id: str,
    step: int,
    model: BitNetModel,
    batch: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    event_name: str
)
```

---

### 3. `src/bitnet/generation.py`
**Purpose:** Text generation utilities for sampling

**Functions:**

```python
def generate_greedy(
    model: BitNetModel,
    input_ids: torch.Tensor,
    max_length: int,
    eos_token_id: int
) -> torch.Tensor

def generate_typical(
    model: BitNetModel,
    input_ids: torch.Tensor,
    max_length: int,
    temperature: float = 0.9,
    top_p: float = 0.95,
    eos_token_id: int = None
) -> torch.Tensor

def generate_stress(
    model: BitNetModel,
    input_ids: torch.Tensor,
    max_length: int,
    temperature: float = 1.5,
    top_k: int = 10,
    eos_token_id: int = None
) -> torch.Tensor

def generate_samples(
    run_id: str,
    step: int,
    model: BitNetModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    device: torch.device
)
```

---

### 4. `train_bitnet_95m.py`
**Purpose:** Main training script

**Structure:**

```python
def main():
    # 1. Setup
    #    - Parse args (run_id, resume_from, etc.)
    #    - Set seeds
    #    - Create run directory

    # 2. Load tokenizer (LLaMA-2)

    # 3. Create model (95M config)

    # 4. Create dataloaders (train + eval)

    # 5. Create optimizer, loss_fn, schedulers

    # 6. Write provenance, config, fingerprint

    # 7. Training loop:
    #    - Log scalars every step
    #    - Checkpoint every 5k steps + mandatory at 29999, 30000
    #    - Eval every 1k steps
    #    - Generate samples every 1k steps
    #    - Log BitNet metrics every 1k steps
    #    - Anomaly detection

    # 8. Final checkpoint and summary

if __name__ == "__main__":
    main()
```

**Command-line args:**
```bash
python train_bitnet_95m.py \
    --run-id bitnet_95M_fineweb_T1.97B_<timestamp> \
    --resume-from runs/previous_run/checkpoints/step_30000/checkpoint.pt \
    --device cuda \
    --no-eval  # Optional: skip eval for faster training
```

---

### 5. `check_training_status.py`
**Purpose:** Monitor training progress without interrupting

**Features:**
- Read latest metrics from scalars.jsonl
- Show current step, loss, LR, WD
- Show tokens/sec throughput
- Show ETA to completion
- Show recent eval loss (if available)
- Show latest sample outputs
- Plot loss curve (optional, if matplotlib available)

**Usage:**
```bash
python check_training_status.py runs/bitnet_95M_fineweb_T1.97B_1703097600
```

**Output:**
```
Training Status: bitnet_95M_fineweb_T1.97B_1703097600
═══════════════════════════════════════════════════════

Progress: 15,234 / 60,000 steps (25.4%)
Tokens:   15.6B / 1.97B

Current Metrics (Step 15234):
  Loss:        2.3456
  LR:          0.001234
  WD:          0.1
  Grad Norm:   0.8234
  Throughput:  12,345 tokens/sec
  Step Time:   234ms

Stage: 1 (Regularization phase)
ETA:  3.2 hours

Last Eval (Step 15000):
  Eval Loss:       2.3123
  Train Loss:      2.3456
  Train-Eval Gap:  0.0333

Recent Samples (Step 15000):
  Prompt: "The quick brown fox"
  Greedy: "The quick brown fox jumped over the lazy dog"
  ...
```

---

## Implementation Order

1. **✅ Already done:**
   - Tied embeddings in transformer.py
   - .gitignore updated

2. **Phase 1: Data & Core Utilities (30 min)**
   - [ ] Create `src/bitnet/data_fineweb.py`
   - [ ] Create `src/bitnet/generation.py`
   - [ ] Test both modules independently

3. **Phase 2: Instrumentation (45 min)**
   - [ ] Create `src/bitnet/instrumentation.py`
   - [ ] Implement provenance functions
   - [ ] Implement checkpoint functions
   - [ ] Implement logging functions
   - [ ] Implement anomaly detection
   - [ ] Test with mock data

4. **Phase 3: Main Training Script (30 min)**
   - [ ] Create `train_bitnet_95m.py`
   - [ ] Implement argparse
   - [ ] Implement main training loop
   - [ ] Wire up all instrumentation
   - [ ] Add progress bars (tqdm)

5. **Phase 4: Status Checker (20 min)**
   - [ ] Create `check_training_status.py`
   - [ ] Implement metrics reading
   - [ ] Implement status display
   - [ ] Add optional plotting

6. **Phase 5: Testing (30 min)**
   - [ ] Dry run with 100 steps
   - [ ] Verify all artifacts created correctly
   - [ ] Test resume functionality
   - [ ] Test status checker

---

## Key Design Decisions

### ✅ Use streaming for FineWeb-Edu
- No need to download 1.3T token corpus
- Just stream what we need (~2B tokens)

### ✅ Modular instrumentation
- Easy to enable/disable components
- Easy to extend with new metrics

### ✅ Resume-friendly checkpoints
- Full RNG state saved
- Can resume from any checkpoint
- Mandatory checkpoints at stage boundaries

### ✅ Non-intrusive status checking
- Read-only access to run directory
- No locks or coordination needed
- Can be run from another terminal

### ✅ Clear separation of concerns
- Data loading: data_fineweb.py
- Generation: generation.py
- Logging/monitoring: instrumentation.py
- Training logic: train_bitnet_95m.py
- Status: check_training_status.py

---

## Testing Strategy

1. **Unit tests** (optional but recommended):
   - Test dataloader yields correct shapes
   - Test generation functions produce valid output
   - Test checkpoint save/load roundtrip

2. **Integration test** (required):
   - Run training for 100 steps
   - Verify directory structure created
   - Verify all files written correctly
   - Verify status checker works
   - Resume from step 50, run 50 more steps
   - Verify deterministic reproduction

3. **Pilot run** (before full training):
   - Run for 1,000 steps (~5 minutes)
   - Check all metrics look reasonable
   - Verify no memory leaks
   - Verify throughput is acceptable

---

## Success Criteria

✅ Training script runs without errors
✅ All instrumentation files created in runs/ directory
✅ Checkpoints can be loaded and resumed
✅ Status checker shows live progress
✅ Git repository stays clean (no artifacts committed)
✅ Generated samples show model learning over time
✅ Full 60k step run completes in ~4-5 hours

---

## Next Steps

Ready to proceed? I'll start with:
1. Create `data_fineweb.py`
2. Create `generation.py`
3. Create `instrumentation.py`
4. Create `train_bitnet_95m.py`
5. Create `check_training_status.py`

Then we'll do a 100-step dry run to verify everything works.
