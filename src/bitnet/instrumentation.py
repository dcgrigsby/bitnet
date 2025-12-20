"""Instrumentation utilities for training: logging, checkpointing, monitoring."""

import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from bitnet.config import BitNetConfig
from bitnet.data_fineweb import FineWebEduDataLoader
from bitnet.generation import generate_greedy, generate_stress, generate_typical
from bitnet.linear import BitLinear
from bitnet.train import TwoStageLRScheduler, TwoStageWDScheduler
from bitnet.transformer import BitNetModel


# =============================================================================
# Directory Setup & Provenance
# =============================================================================


def create_run_directory(run_id: str) -> Path:
    """Create run directory structure.

    Args:
        run_id: Unique run identifier

    Returns:
        Path to run directory
    """
    run_dir = Path("runs") / run_id

    # Create subdirectories
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "anomalies").mkdir(parents=True, exist_ok=True)

    return run_dir


def write_provenance(run_id: str, config: BitNetConfig) -> None:
    """Write provenance information for reproducibility.

    Args:
        run_id: Unique run identifier
        config: Model configuration
    """
    run_dir = Path("runs") / run_id

    # Get git info
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        git_dirty = (
            subprocess.check_output(["git", "status", "--porcelain"])
            .decode("utf-8")
            .strip()
            != ""
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"
        git_dirty = False

    # Collect provenance
    provenance = {
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "seeds": {
            "python": random.getstate()[1][0],  # First element of MT state
            "numpy": int(np.random.get_state()[1][0]),  # First element of state
            "torch": torch.initial_seed(),
        },
    }

    # Write to file
    with open(run_dir / "meta" / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)


def write_config(
    run_id: str, config: BitNetConfig, training_args: dict[str, Any]
) -> None:
    """Write model and training configuration.

    Args:
        run_id: Unique run identifier
        config: Model configuration
        training_args: Training hyperparameters
    """
    run_dir = Path("runs") / run_id

    # Build config dict
    config_dict = {
        "model": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "ffn_hidden_size": config.ffn_hidden_size,
            "max_seq_length": config.max_seq_length,
            "norm_eps": config.norm_eps,
            "head_dim": config.head_dim,
            "embeddings_tied": True,  # Now implemented
        },
        "training": training_args,
    }

    # Write to file
    with open(run_dir / "meta" / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def write_dataset_fingerprint(
    run_id: str,
    dataloader: FineWebEduDataLoader,
    tokenizer_name: str,
) -> None:
    """Write dataset fingerprint for verification.

    Args:
        run_id: Unique run identifier
        dataloader: Data loader instance (must have been iterated at least once)
        tokenizer_name: Name/path of tokenizer used
    """
    run_dir = Path("runs") / run_id

    fingerprint = {
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "dataset_config": dataloader.name,
        "split": dataloader.split,
        "tokenizer": tokenizer_name,
        "first_1000_token_ids": dataloader.get_fingerprint_tokens(),
        "first_1000_token_ids_sha256": dataloader.get_fingerprint(),
    }

    with open(run_dir / "meta" / "dataset_fingerprint.json", "w") as f:
        json.dump(fingerprint, f, indent=2)


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(
    run_id: str,
    step: int,
    model: BitNetModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: TwoStageLRScheduler,
    wd_scheduler: TwoStageWDScheduler,
    loss: float,
    config: BitNetConfig,
    mandatory: bool = False,
) -> None:
    """Save training checkpoint.

    Args:
        run_id: Unique run identifier
        step: Current training step
        model: Model instance
        optimizer: Optimizer instance
        lr_scheduler: LR scheduler instance
        wd_scheduler: WD scheduler instance
        loss: Current loss value
        config: Model configuration
        mandatory: Whether this is a mandatory checkpoint (stage boundary)
    """
    run_dir = Path("runs") / run_id
    ckpt_dir = run_dir / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "tokens_seen": step * config.max_seq_length * optimizer.param_groups[0].get("batch_size", 32),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state": {
            "current_step": lr_scheduler.current_step,
            "total_steps": lr_scheduler.total_steps,
            "warmup_steps": lr_scheduler.warmup_steps,
            "peak_lr": lr_scheduler.peak_lr,
            "stage_boundary": lr_scheduler.stage_boundary,
            "stage2_lr": lr_scheduler.stage2_lr,
            "final_lr": lr_scheduler.final_lr,
        },
        "wd_scheduler_state": {
            "current_step": wd_scheduler.current_step,
            "total_steps": wd_scheduler.total_steps,
            "stage1_steps": wd_scheduler.stage1_steps,
            "stage1_wd": wd_scheduler.stage1_wd,
            "stage2_wd": wd_scheduler.stage2_wd,
        },
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        "loss": loss,
        "mandatory": mandatory,
    }

    torch.save(checkpoint, ckpt_dir / "checkpoint.pt")


def load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """Load checkpoint from file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Checkpoint dictionary
    """
    return torch.load(checkpoint_path, map_location="cpu")


# =============================================================================
# Metrics Logging
# =============================================================================


def log_scalars(
    run_id: str,
    step: int,
    loss: float,
    lr: float,
    wd: float,
    grad_norm: float,
    step_time_ms: float,
    batch_size: int,
    seq_len: int,
    **kwargs: Any,
) -> None:
    """Log scalar metrics to JSONL file.

    Args:
        run_id: Unique run identifier
        step: Current training step
        loss: Training loss
        lr: Learning rate
        wd: Weight decay
        grad_norm: Gradient norm
        step_time_ms: Step time in milliseconds
        batch_size: Batch size
        seq_len: Sequence length
        **kwargs: Additional metrics to log
    """
    run_dir = Path("runs") / run_id

    tokens_per_sec = (batch_size * seq_len) / (step_time_ms / 1000)

    metrics = {
        "step": step,
        "tokens_seen": step * batch_size * seq_len,
        "loss": loss,
        "lr": lr,
        "wd": wd,
        "grad_norm": grad_norm,
        "step_time_ms": step_time_ms,
        "tokens_per_sec": tokens_per_sec,
    }

    # Add GPU metrics if available
    if torch.cuda.is_available():
        metrics["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        metrics["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

    # Add any additional metrics
    metrics.update(kwargs)

    # Append to JSONL file
    with open(run_dir / "metrics" / "scalars.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")


def log_bitnet_metrics(run_id: str, step: int, model: BitNetModel) -> None:
    """Log BitNet-specific quantization metrics.

    Args:
        run_id: Unique run identifier
        step: Current training step
        model: Model instance
    """
    run_dir = Path("runs") / run_id

    metrics = {"step": step}

    # Per-layer quantization distribution
    for layer_idx, block in enumerate(model.blocks):
        # Attention QKV projection
        if hasattr(block.attention, "qkv_proj") and isinstance(
            block.attention.qkv_proj, BitLinear
        ):
            qkv_weights = block.attention.qkv_proj.weight
            # Note: BitLinear quantizes on forward pass, so we approximate
            # by quantizing the current weights
            with torch.no_grad():
                scale = qkv_weights.abs().mean()
                if scale > 0:
                    qkv_quant = torch.clamp(
                        torch.round(qkv_weights / scale), -1, 1
                    )
                else:
                    qkv_quant = torch.zeros_like(qkv_weights)

                metrics[f"L{layer_idx}_attn_qkv_neg1_pct"] = (
                    (qkv_quant == -1).float().mean().item()
                )
                metrics[f"L{layer_idx}_attn_qkv_zero_pct"] = (
                    (qkv_quant == 0).float().mean().item()
                )
                metrics[f"L{layer_idx}_attn_qkv_pos1_pct"] = (
                    (qkv_quant == +1).float().mean().item()
                )

        # FFN gate_up projection
        if hasattr(block.feedforward, "gate_up") and isinstance(
            block.feedforward.gate_up, BitLinear
        ):
            gate_up_weights = block.feedforward.gate_up.weight
            with torch.no_grad():
                scale = gate_up_weights.abs().mean()
                if scale > 0:
                    gate_up_quant = torch.clamp(
                        torch.round(gate_up_weights / scale), -1, 1
                    )
                else:
                    gate_up_quant = torch.zeros_like(gate_up_weights)

                metrics[f"L{layer_idx}_ffn_gate_up_neg1_pct"] = (
                    (gate_up_quant == -1).float().mean().item()
                )
                metrics[f"L{layer_idx}_ffn_gate_up_zero_pct"] = (
                    (gate_up_quant == 0).float().mean().item()
                )
                metrics[f"L{layer_idx}_ffn_gate_up_pos1_pct"] = (
                    (gate_up_quant == +1).float().mean().item()
                )

    # Append to JSONL
    with open(run_dir / "metrics" / "scalars.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")


# =============================================================================
# Evaluation
# =============================================================================


def run_evaluation(
    run_id: str,
    step: int,
    model: BitNetModel,
    eval_dataloader: Any,
    loss_fn: nn.Module,
    device: torch.device,
    train_loss_ma: float,
) -> float:
    """Run evaluation on frozen eval set.

    Args:
        run_id: Unique run identifier
        step: Current training step
        model: Model instance
        eval_dataloader: Evaluation data loader
        loss_fn: Loss function
        device: Device to run on
        train_loss_ma: Training loss moving average

    Returns:
        Evaluation loss
    """
    run_dir = Path("runs") / run_id

    model.eval()
    eval_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = batch.to(device)
            logits = model(batch)

            # Compute loss
            flat_logits = logits[:, :-1, :].reshape(-1, model.config.vocab_size)
            flat_targets = batch[:, 1:].reshape(-1)
            loss = loss_fn(flat_logits, flat_targets)

            eval_loss += loss.item()
            num_batches += 1

    eval_loss /= num_batches if num_batches > 0 else 1

    # Log to file
    with open(run_dir / "eval" / "eval_results.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "step": step,
                    "eval_loss": eval_loss,
                    "train_loss": train_loss_ma,
                    "train_eval_gap": train_loss_ma - eval_loss,
                }
            )
            + "\n"
        )

    model.train()
    return eval_loss


# =============================================================================
# Sampling
# =============================================================================

FIXED_PROMPTS = [
    "The quick brown fox",
    "In the beginning",
    "Once upon a time",
    "The capital of France",
    "Machine learning is",
]


def generate_samples(
    run_id: str,
    step: int,
    model: BitNetModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_length: int = 50,
) -> None:
    """Generate and log text samples from fixed prompts.

    Args:
        run_id: Unique run identifier
        step: Current training step
        model: Model instance
        tokenizer: Tokenizer instance
        device: Device to run on
        max_length: Maximum generation length
    """
    run_dir = Path("runs") / run_id

    model.eval()

    for prompt_idx, prompt in enumerate(FIXED_PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        input_len = input_ids.shape[1]

        for mode in ["greedy", "typical", "stress"]:
            # Fixed seed per (prompt, step, mode) for reproducibility
            seed = hash((prompt_idx, step, mode)) % (2**32)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Generate based on mode
            with torch.no_grad():
                if mode == "greedy":
                    generated_ids = generate_greedy(
                        model, input_ids, max_length, tokenizer.eos_token_id
                    )
                elif mode == "typical":
                    generated_ids = generate_typical(
                        model,
                        input_ids,
                        max_length,
                        temperature=0.9,
                        top_p=0.95,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                else:  # stress
                    generated_ids = generate_stress(
                        model,
                        input_ids,
                        max_length,
                        temperature=1.5,
                        top_k=10,
                        eos_token_id=tokenizer.eos_token_id,
                    )

            # Decode
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Compute metrics
            gen_tokens = generated_ids[0, input_len:].tolist()
            token_counts = Counter(gen_tokens)
            max_repetition = max(token_counts.values()) if token_counts else 0
            unique_tokens = len(token_counts)

            # Compute entropies for first 10 positions
            entropies = []
            top5_sequences = []
            with torch.no_grad():
                for pos in range(
                    input_len, min(input_len + 10, generated_ids.shape[1])
                ):
                    logits = model(generated_ids[:, :pos])
                    probs = F.softmax(logits[0, -1, :], dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                    entropies.append(entropy)

                    top5_probs, top5_ids = torch.topk(probs, k=5)
                    top5_sequences.append(
                        {
                            "pos": pos - input_len,
                            "entropy": entropy,
                            "top5_tokens": [
                                tokenizer.decode([tid]) for tid in top5_ids.tolist()
                            ],
                            "top5_probs": top5_probs.tolist(),
                        }
                    )

            # Log
            with open(run_dir / "samples" / "samples.jsonl", "a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "prompt": prompt,
                            "mode": mode,
                            "seed": seed,
                            "generated_text": generated_text,
                            "generated_ids": generated_ids[0].tolist(),
                            "mean_entropy": (
                                sum(entropies) / len(entropies) if entropies else 0.0
                            ),
                            "max_repetition": max_repetition,
                            "unique_tokens": unique_tokens,
                            "total_tokens": len(gen_tokens),
                            "logits_first_10": top5_sequences,
                        }
                    )
                    + "\n"
                )

    model.train()


# =============================================================================
# Anomaly Detection
# =============================================================================


def detect_anomaly(
    loss: float,
    loss_history: list[float],
    mean_entropy: float = 0.0,
    max_repetition: int = 0,
) -> tuple[bool, str | None]:
    """Detect training anomalies.

    Args:
        loss: Current loss value
        loss_history: Recent loss values (for moving average)
        mean_entropy: Mean entropy from recent samples
        max_repetition: Max repetition count from recent samples

    Returns:
        (is_anomaly, event_name) tuple
    """
    # NaN/Inf detection
    if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
        return True, "nan_inf_loss"

    # Loss spike (>2x moving average)
    if len(loss_history) >= 100:
        moving_avg = sum(loss_history[-100:]) / 100
        if loss > 2 * moving_avg:
            return True, "loss_spike"

    # Entropy collapse
    if mean_entropy > 0 and mean_entropy < 1.0:
        return True, "entropy_collapse"

    # Repetition surge
    if max_repetition > 20:
        return True, "repetition_surge"

    return False, None


def trigger_anomaly_dump(
    run_id: str,
    step: int,
    model: BitNetModel,
    batch: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    event_name: str,
) -> None:
    """Trigger anomaly debug dump.

    Args:
        run_id: Unique run identifier
        step: Current training step
        model: Model instance
        batch: Current batch
        tokenizer: Tokenizer instance
        event_name: Name of anomaly event
    """
    run_dir = Path("runs") / run_id
    dump_dir = run_dir / "anomalies" / f"step_{step:06d}_{event_name}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Forward pass
    with torch.no_grad():
        logits = model(batch)

        # Save top-k distributions for fixed prompts
        for prompt in FIXED_PROMPTS[:3]:  # First 3 prompts only
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(batch.device)
            logits = model(input_ids)
            probs = F.softmax(logits[0, -1, :], dim=-1)

            top20_probs, top20_ids = torch.topk(probs, k=20)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

            with open(
                dump_dir / f'prompt_{prompt[:20].replace(" ", "_")}_top20.json', "w"
            ) as f:
                json.dump(
                    {
                        "prompt": prompt,
                        "tokens": [tokenizer.decode([tid]) for tid in top20_ids.tolist()],
                        "probs": top20_probs.tolist(),
                        "entropy": entropy,
                    },
                    f,
                    indent=2,
                )

    model.train()

    print(f"[ANOMALY] Debug artifacts dumped to {dump_dir}")
