#!/usr/bin/env python3
"""
Train BitNet b1.58 95M parameter model on FineWeb-Edu.

Compute-optimal configuration per Chinchilla scaling:
- 95M parameters
- 1.97B tokens (T/P ≈ 20.6)
- FineWeb-Edu dataset (1.3T tokens, zero repetition)
- LLaMA-2 tokenizer (32k vocab)
- Tied embeddings
- Two-stage LR/WD schedule

Full instrumentation with checkpoints, metrics, samples, and anomaly detection.
All artifacts stored in runs/<run_id>/ directory (git-clean).
"""

import argparse
import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import LlamaTokenizer, PreTrainedTokenizer
from tqdm import tqdm

from bitnet.config import BitNetConfig
from bitnet.data_fineweb import FineWebEduDataLoader
from bitnet.instrumentation import (
    create_run_directory,
    detect_anomaly,
    generate_samples,
    log_bitnet_metrics,
    log_scalars,
    run_evaluation,
    save_checkpoint,
    trigger_anomaly_dump,
    write_config,
    write_dataset_fingerprint,
    write_provenance,
)
from bitnet.train import TwoStageLRScheduler, TwoStageWDScheduler
from bitnet.transformer import BitNetModel


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train BitNet 95M model on FineWeb-Edu"
    )

    # Run configuration
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (default: auto-generated from timestamp)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # Training configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Global batch size (default: 32)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length (default: 1024)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=60000,
        help="Total training steps (default: 60000)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )

    # Logging configuration
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log scalars every N steps (default: 100)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Run evaluation every N steps (default: 1000)",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=1000,
        help="Generate samples every N steps (default: 1000)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Save checkpoint every N steps (default: 5000)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation (faster training)",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Generate run ID if not provided
    if args.run_id is None:
        timestamp = int(time.time())
        args.run_id = f"bitnet_95M_fineweb_T1.97B_{timestamp}"

    print("=" * 80)
    print(f"BitNet 95M Training Run: {args.run_id}")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Total steps: {args.num_steps}")
    print(f"Total tokens: {args.num_steps * args.batch_size * args.seq_len:,}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    print()

    # Create run directory
    create_run_directory(args.run_id)

    # Load tokenizer (LLaMA-2)
    print("Loading LLaMA-2 tokenizer...")
    tokenizer = cast(
        PreTrainedTokenizer,
        LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    print()

    # Create model configuration (95M params)
    config = BitNetConfig(
        vocab_size=32000,  # LLaMA-2 tokenizer
        hidden_size=768,
        num_layers=8,
        num_heads=12,
        num_kv_heads=6,  # GQA 2:1
        ffn_hidden_size=3072,  # 4x expansion
        max_seq_length=args.seq_len,
        norm_eps=1e-5,
        learning_rate=0.0015,  # Peak LR
        weight_decay=0.1,  # Stage 1
        warmup_steps=375,
        adam_beta1=0.9,
        adam_beta2=0.95,
    )

    # Create model
    device = torch.device(args.device)
    model = BitNetModel(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {num_params:,} parameters")
    print(f"Target: ~95M parameters")
    print(f"Embeddings tied: Yes")
    print()

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader = FineWebEduDataLoader(
        tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        split="train",
    )

    eval_dataloader = None
    if not args.no_eval:
        eval_dataloader = FineWebEduDataLoader(
            tokenizer,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_steps=100,  # Fixed eval set size
            split="train",  # Use train split but different iteration
        )

    print("Dataloaders created.")
    print()

    # Create optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=cast(int, tokenizer.pad_token_id))

    # Create schedulers
    lr_scheduler = TwoStageLRScheduler(optimizer, config, args.num_steps)
    wd_scheduler = TwoStageWDScheduler(optimizer, args.num_steps)

    # Write provenance and configuration
    print("Writing provenance and configuration...")
    write_provenance(args.run_id, config)

    training_args = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "total_steps": args.num_steps,
        "total_tokens": args.num_steps * args.batch_size * args.seq_len,
        "grad_accum_steps": args.grad_accum_steps,
        "peak_lr": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "stage1_wd": 0.1,
        "stage2_wd": 0.0,
        "grad_clip_norm": 1.0,
        "adam_beta1": config.adam_beta1,
        "adam_beta2": config.adam_beta2,
        "dataset": "HuggingFaceFW/fineweb-edu",
        "tokenizer": "meta-llama/Llama-2-7b-hf",
    }

    write_config(args.run_id, config, training_args)
    print("Provenance written.")
    print()

    # Training loop
    print("Starting training...")
    print("=" * 80)
    print()

    model.train()
    step = 0
    loss_history: list[float] = []
    train_loss_ma = 0.0  # Moving average

    # Progress bar
    pbar = tqdm(total=args.num_steps, desc="Training", unit="step")

    # Iterate through data
    for batch_idx, batch in enumerate(train_dataloader):
        batch = batch.to(device)

        # Write dataset fingerprint after first batch
        if batch_idx == 0:
            write_dataset_fingerprint(
                args.run_id, train_dataloader, "meta-llama/Llama-2-7b-hf"
            )

        # Start timing
        step_start = time.time()

        # Forward pass
        optimizer.zero_grad()
        logits = model(batch)

        # Compute loss
        flat_logits = logits[:, :-1, :].reshape(-1, config.vocab_size)
        flat_targets = batch[:, 1:].reshape(-1)
        loss = loss_fn(flat_logits, flat_targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        optimizer.step()
        lr_scheduler.step()
        wd_scheduler.step()

        # End timing
        step_time_ms = (time.time() - step_start) * 1000

        step += 1

        # Update loss history
        loss_history.append(loss.item())
        if len(loss_history) > 1000:
            loss_history = loss_history[-1000:]  # Keep last 1000

        train_loss_ma = sum(loss_history[-100:]) / min(100, len(loss_history))

        # Log scalars
        if step % args.log_interval == 0 or step == 1:
            current_lr = cast(float, optimizer.param_groups[0]["lr"])
            current_wd = cast(float, optimizer.param_groups[0]["weight_decay"])

            log_scalars(
                args.run_id,
                step,
                loss.item(),
                current_lr,
                current_wd,
                grad_norm.item(),
                step_time_ms,
                args.batch_size,
                args.seq_len,
            )

            # Update progress bar
            stage = "Stage 2" if step > args.num_steps // 2 else "Stage 1"
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.6f}",
                    "wd": f"{current_wd:.2f}",
                    "stage": stage,
                }
            )

        # Log BitNet metrics
        if step % args.log_interval == 0:
            log_bitnet_metrics(args.run_id, step, model)

        # Evaluation
        if not args.no_eval and step % args.eval_interval == 0 and eval_dataloader:
            eval_loss = run_evaluation(
                args.run_id,
                step,
                model,
                eval_dataloader,
                loss_fn,
                device,
                train_loss_ma,
            )
            print(f"\n[Step {step}] Eval loss: {eval_loss:.4f}\n")

        # Generate samples
        if step % args.sample_interval == 0:
            generate_samples(args.run_id, step, model, tokenizer, device)

        # Checkpointing
        mandatory = step in [29999, 30000]  # Stage boundaries
        if step % args.checkpoint_interval == 0 or mandatory:
            save_checkpoint(
                args.run_id,
                step,
                model,
                optimizer,
                lr_scheduler,
                wd_scheduler,
                loss.item(),
                config,
                mandatory=mandatory,
            )
            if mandatory:
                print(f"\n[MANDATORY CHECKPOINT] Saved at step {step}\n")

        # Anomaly detection
        mean_entropy = 0.0  # Could compute from recent samples
        max_repetition = 0  # Could compute from recent samples

        is_anomaly, event_name = detect_anomaly(
            loss.item(), loss_history, mean_entropy, max_repetition
        )

        if is_anomaly and event_name:
            print(f"\n[ANOMALY DETECTED] {event_name} at step {step}\n")
            trigger_anomaly_dump(
                args.run_id, step, model, batch, tokenizer, event_name
            )

        pbar.update(1)

    pbar.close()

    # Final checkpoint
    print("\nSaving final checkpoint...")
    save_checkpoint(
        args.run_id,
        args.num_steps,
        model,
        optimizer,
        lr_scheduler,
        wd_scheduler,
        loss.item(),
        config,
        mandatory=True,
    )

    print("\n" + "=" * 80)
    print(f"Training complete!")
    print(f"Run ID: {args.run_id}")
    print(f"Run directory: runs/{args.run_id}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
