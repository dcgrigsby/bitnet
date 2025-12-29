#!/usr/bin/env python3
"""
Train BitNet b1.58 5M parameter model on synthetic arithmetic.

Ultra-fast validation configuration:
- 5M parameters (ultra-tiny model)
- 50M tokens (T/P ≈ 10)
- Synthetic arithmetic dataset (X + Y = Z patterns)
- Character-level tokenizer (32 vocab size)
- Expected runtime: ~30 minutes on RTX 3060

Purpose: Quick sanity check that BitNet quantization and gradient flow work.
If this fails, there are bugs. If this works, move to TinyStories.
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
from tqdm import tqdm

from bitnet.config_tiny import BitNetConfigArithmetic
from bitnet.data_arithmetic import ArithmeticDataLoader, ArithmeticTokenizer
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
        description="Train BitNet 5M arithmetic model"
    )

    # Run configuration
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (default: auto-generated)",
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
        default=64,
        help="Sequence length (default: 64)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10000,
        help="Total training steps (default: 10000)",
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
        default=500,
        help="Generate samples every N steps (default: 500)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2000,
        help="Save checkpoint every N steps (default: 2000)",
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


def test_arithmetic(
    run_id: str,
    step: int,
    model: BitNetModel,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
) -> None:
    """Test model on arithmetic problems."""
    import json
    run_dir = Path("runs") / run_id

    model.eval()

    # Test problems
    test_problems = [
        "5 + 3 =",
        "12 - 7 =",
        "8 * 6 =",
        "20 / 4 =",
        "15 + 25 =",
    ]

    results = []

    with torch.no_grad():
        for problem in test_problems:
            input_ids = torch.tensor([tokenizer.encode(problem)]).to(device)

            # Generate answer (should be 1-3 digits)
            generated = input_ids.clone()
            space_count = 0
            for _ in range(10):  # Max 10 tokens for answer
                logits = model(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                # Stop at second space (after answer digits)
                if next_token.item() == tokenizer.char_to_id[" "]:
                    space_count += 1
                    if space_count == 2:
                        break

            full_text = tokenizer.decode(generated[0].tolist())
            results.append({
                "problem": problem,
                "completion": full_text,
            })

            print(f"  {full_text}")

    # Save results
    with open(run_dir / "samples" / "samples.jsonl", "a") as f:
        f.write(json.dumps({
            "step": step,
            "test_results": results,
        }) + "\n")

    model.train()


def main() -> None:
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)

    # Generate run ID
    if args.run_id is None:
        timestamp = int(time.time())
        args.run_id = f"bitnet_5M_arithmetic_{timestamp}"

    print("=" * 80)
    print(f"BitNet 5M Arithmetic Training: {args.run_id}")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Total steps: {args.num_steps}")
    print(f"Total tokens: {args.num_steps * args.batch_size * args.seq_len:,}")
    print("=" * 80)
    print()

    # Create run directory
    create_run_directory(args.run_id)

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = ArithmeticTokenizer()
    print(f"Tokenizer created: vocab_size={tokenizer.vocab_size}")
    print()

    # Create config
    config = BitNetConfigArithmetic(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        ffn_hidden_size=768,
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
    dataloader = ArithmeticDataLoader(
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
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

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

        # Test arithmetic
        if step % args.sample_interval == 0:
            print(f"\n[Step {step}] Testing arithmetic...")
            test_arithmetic(args.run_id, step, model, tokenizer, device)

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

    # Final test
    print("\n" + "=" * 80)
    print("Final arithmetic test:")
    print("=" * 80)
    test_arithmetic(args.run_id, args.num_steps, model, tokenizer, device)

    print("\n" + "=" * 80)
    print(f"Training complete!")
    print(f"Run ID: {args.run_id}")
    print(f"Final loss: {loss_history[-1] if loss_history else 0:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
