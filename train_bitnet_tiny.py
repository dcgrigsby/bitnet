#!/usr/bin/env python3
"""
Train BitNet b1.58 12M parameter model on TinyStories.

Infrastructure validation configuration:
- 12M parameters (tiny model)
- 240M tokens (T/P ≈ 20, Chinchilla-optimal)
- TinyStories dataset (simple language)
- Custom 2K vocabulary tokenizer
- Expected runtime: ~1.5 hours on RTX 3060

Purpose: Validate that BitNet quantization, gradients, and training work correctly
on a simple task before scaling to larger models.
"""

import argparse
import random
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer
from tqdm import tqdm

from bitnet.config_tiny import BitNetConfigTiny
from bitnet.data_tinystories import TinyStoriesDataLoader
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
        description="Train BitNet 12M tiny model on TinyStories"
    )

    # Run configuration
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (default: auto-generated)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tinystories_tokenizer.json",
        help="Path to TinyStories tokenizer (default: tinystories_tokenizer.json)",
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
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length (default: 256)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=30000,
        help="Total training steps (default: 30000)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )

    # Logging
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log every N steps (default: 100)",
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

    return parser.parse_args()


def create_run_directory(run_id: str) -> Path:
    """Create run directory structure."""
    run_dir = Path("runs") / run_id
    (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    return run_dir


def log_metrics(run_id: str, step: int, metrics: dict[str, Any]) -> None:
    """Log metrics to JSONL file."""
    import json
    run_dir = Path("runs") / run_id
    with open(run_dir / "metrics" / "scalars.jsonl", "a") as f:
        metrics["step"] = step
        f.write(json.dumps(metrics) + "\n")


def generate_samples(
    run_id: str,
    step: int,
    model: BitNetModel,
    tokenizer: Tokenizer,
    device: torch.device,
) -> None:
    """Generate text samples from the model."""
    import json
    run_dir = Path("runs") / run_id

    model.eval()
    prompts = [
        "Once upon a time",
        "There was a little",
        "One day a boy",
    ]

    with torch.no_grad():
        for prompt in prompts:
            input_ids = torch.tensor([tokenizer.encode(prompt).ids]).to(device)

            # Simple greedy generation
            generated = input_ids.clone()
            for _ in range(50):
                logits = model(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if we hit EOS or exceed length
                if next_token.item() == tokenizer.token_to_id("</s>"):
                    break

            generated_text = tokenizer.decode(generated[0].tolist())

            with open(run_dir / "samples" / "samples.jsonl", "a") as f:
                f.write(json.dumps({
                    "step": step,
                    "prompt": prompt,
                    "generated_text": generated_text,
                }) + "\n")

            print(f"  [{prompt}] → {generated_text}")

    model.train()


def main() -> None:
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)

    # Generate run ID
    if args.run_id is None:
        timestamp = int(time.time())
        args.run_id = f"bitnet_12M_tinystories_{timestamp}"

    print("=" * 80)
    print(f"BitNet 12M TinyStories Training: {args.run_id}")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Total steps: {args.num_steps}")
    print(f"Total tokens: {args.num_steps * args.batch_size * args.seq_len:,}")
    print(f"Tokenizer: {args.tokenizer}")
    print("=" * 80)
    print()

    # Create run directory
    create_run_directory(args.run_id)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")
    print()

    # Create config
    config = BitNetConfigTiny(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=384,
        num_layers=6,
        num_heads=8,
        num_kv_heads=4,
        ffn_hidden_size=1152,
        max_seq_length=args.seq_len,
    )

    # Create model
    device = torch.device(args.device)
    model = BitNetModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {num_params:,} parameters")
    print()

    # Create dataloader
    print("Creating dataloader...")
    dataloader = TinyStoriesDataLoader(
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
    )
    print("Dataloader created.")
    print()

    # Create optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

    # Create schedulers
    lr_scheduler = TwoStageLRScheduler(optimizer, config, args.num_steps)
    wd_scheduler = TwoStageWDScheduler(optimizer, args.num_steps)

    # Training loop
    print("Starting training...")
    print("=" * 80)
    print()

    model.train()
    step = 0
    loss_history: list[float] = []
    pbar = tqdm(total=args.num_steps, desc="Training", unit="step")

    for batch in dataloader:
        batch = batch.to(device)
        step_start = time.time()

        # Forward pass
        logits = model(batch)

        # Compute loss
        flat_logits = logits[:, :-1, :].reshape(-1, config.vocab_size)
        flat_targets = batch[:, 1:].reshape(-1)
        loss = loss_fn(flat_logits, flat_targets)

        # Backward pass
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        wd_scheduler.step()
        optimizer.zero_grad()

        step += 1
        step_time_ms = (time.time() - step_start) * 1000
        loss_value = loss.item()
        loss_history.append(loss_value)

        # Logging
        if step % args.log_interval == 0 or step == 1:
            current_lr = cast(float, optimizer.param_groups[0]["lr"])
            current_wd = cast(float, optimizer.param_groups[0]["weight_decay"])

            metrics = {
                "loss": loss_value,
                "lr": current_lr,
                "wd": current_wd,
                "grad_norm": grad_norm.item(),
                "step_time_ms": step_time_ms,
                "tokens_per_sec": (args.batch_size * args.seq_len) / (step_time_ms / 1000),
            }

            if torch.cuda.is_available():
                metrics["gpu_mem_gb"] = torch.cuda.memory_allocated() / 1e9

            log_metrics(args.run_id, step, metrics)

            stage = "Stage 2" if step > args.num_steps // 2 else "Stage 1"
            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "lr": f"{current_lr:.6f}",
                "stage": stage,
            })

        # Generate samples
        if step % args.sample_interval == 0:
            print(f"\n[Step {step}] Generating samples...")
            generate_samples(args.run_id, step, model, tokenizer, device)

        # Checkpointing
        if step % args.checkpoint_interval == 0:
            ckpt_dir = Path("runs") / args.run_id / "checkpoints" / f"step_{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_value,
            }, ckpt_dir / "checkpoint.pt")

        pbar.update(1)

    pbar.close()

    # Final checkpoint
    print("\nSaving final checkpoint...")
    ckpt_dir = Path("runs") / args.run_id / "checkpoints" / f"step_{args.num_steps:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": args.num_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_history[-1] if loss_history else 0.0,
    }, ckpt_dir / "checkpoint.pt")

    print("\n" + "=" * 80)
    print(f"Training complete!")
    print(f"Run ID: {args.run_id}")
    print(f"Final loss: {loss_history[-1] if loss_history else 0:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
