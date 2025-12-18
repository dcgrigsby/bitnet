"""
Stage 1 training with reduced learning rate to test LR sensitivity.

Tests hypothesis: LR=0.0015 is too aggressive for 256-hidden model.
This script uses LR=0.0005 (1/3 of standard BitNet peak).
"""
import sys
from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, PreTrainedTokenizer

from bitnet.config import BitNetConfig
from bitnet.data import WikiTextDataLoader
from bitnet.train import TwoStageLRScheduler, TwoStageWDScheduler, train_step
from bitnet.transformer import BitNetModel


def count_unique_tokens(model: BitNetModel, tokenizer: PreTrainedTokenizer, device: torch.device, num_samples: int = 10) -> int:
    """Generate predictions and count unique tokens."""
    model.eval()
    unique_tokens = set()

    with torch.no_grad():
        for i in range(num_samples):
            # Generate random input
            input_ids = torch.randint(0, tokenizer.vocab_size, (1, 8)).to(device)
            logits = model(input_ids)
            predictions = torch.argmax(logits[0, -1, :], dim=-1)
            unique_tokens.add(int(predictions.item()))

    model.train()
    return len(unique_tokens)


def main():
    """Train using reduced LR (0.0005 instead of 0.0015)."""

    # Parse command line arguments
    num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/reduced_lr_{num_steps}.txt"

    print(f"Stage 1 Training with Reduced LR")
    print(f"  Total steps: {num_steps}")
    print(f"  Peak LR: 0.0005 (vs normal 0.0015)")
    print(f"  Output file: {output_file}")
    print()

    # Load tokenizer
    tokenizer = cast(PreTrainedTokenizer, GPT2Tokenizer.from_pretrained("gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    config = BitNetConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=4,
        ffn_hidden_size=512,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BitNetModel(config).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")
    print()

    # Setup training with REDUCED learning rate
    reduced_lr = 0.0005  # 1/3 of standard 0.0015
    optimizer = optim.Adam(
        model.parameters(),
        lr=reduced_lr,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=cast(int, tokenizer.pad_token_id))

    # Create dataloader
    batch_size = 4
    seq_len = 32

    dataloader = WikiTextDataLoader(
        tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=num_steps,
    )

    # Use Stage 1 settings only
    print(f"Training config:")
    print(f"  LR: 0.0005 (reduced from 0.0015)")
    print(f"  WD: 0.1 (Stage 1)")
    print(f"  Warmup: {config.warmup_steps} steps")
    print()

    # Create custom LR scheduler with reduced peak LR
    # We'll use TwoStageLRScheduler but with reduced peak
    class ReducedLRScheduler(TwoStageLRScheduler):
        def __init__(self, optimizer, config, total_steps):
            super().__init__(optimizer, config, total_steps)
            self.peak_lr = 0.0005  # Override peak LR
            self.stage2_lr = self.peak_lr * 2.0 / 3.0
            self.final_lr = self.peak_lr * 0.001

    lr_scheduler = ReducedLRScheduler(optimizer, config, num_steps)
    wd_scheduler = TwoStageWDScheduler(optimizer, num_steps)
    wd_scheduler.stage1_steps = num_steps + 1  # Keep at 0.1 throughout

    _ = model.train()
    step = 0

    with open(output_file, 'w') as f:
        f.write(f"Reduced LR Training - {num_steps} steps (LR=0.0005)\n")
        f.write("Step,Loss,UniquePredictions,LR,WD\n")

        for batch in dataloader:
            batch = batch.to(device)

            loss = train_step(
                model,
                batch,
                optimizer,
                loss_fn,
                lr_scheduler=lr_scheduler,
                wd_scheduler=wd_scheduler,
            )

            step += 1

            # Evaluate every 1000 steps
            if step % 1000 == 0 or step == 1:
                current_lr = cast(float, optimizer.param_groups[0]["lr"])
                current_wd = cast(float, optimizer.param_groups[0]["weight_decay"])
                unique_count = count_unique_tokens(model, tokenizer, device, num_samples=20)

                progress = (step / num_steps) * 100
                print(f"Step {step:6d}/{num_steps} ({progress:5.1f}%): Loss={loss:.4f}, UniqueTokens={unique_count:2d}, LR={current_lr:.6f}, WD={current_wd:.2f}")
                f.write(f"{step},{loss:.4f},{unique_count},{current_lr:.6f},{current_wd:.2f}\n")
                f.flush()

    print(f"\nTraining complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
