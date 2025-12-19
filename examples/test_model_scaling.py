"""
Test BitNet training across different model sizes.

Hypothesis: Collapse is a scaling issue. Smaller models need different hyperparameters.
Test: Train models of different hidden sizes with standard BitNet LR (0.0015) and WD schedule (0.1→0.0).
Measure: Do larger models maintain vocabulary diversity better?

Model sizes:
- 256 hidden (current - baseline, collapses)
- 512 hidden (2x larger)
- 768 hidden (3x larger)
- 1024 hidden (4x larger)
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
            input_ids = torch.randint(0, tokenizer.vocab_size, (1, 8)).to(device)
            logits = model(input_ids)
            predictions = torch.argmax(logits[0, -1, :], dim=-1)
            unique_tokens.add(int(predictions.item()))

    model.train()
    return len(unique_tokens)


def main():
    """Test Stage 1 training with different model sizes."""

    num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    hidden_size = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    output_file = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/model_scale_{hidden_size}_{num_steps}.txt"

    print(f"BitNet Model Scaling Test")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Total steps: {num_steps}")
    print(f"  LR: 0.0015 (standard BitNet)")
    print(f"  WD Stage 1: 0.1 (standard)")
    print(f"  WD Stage 2: 0.0 (per Microsoft papers)")
    print(f"  Output file: {output_file}")
    print()

    tokenizer = cast(PreTrainedTokenizer, GPT2Tokenizer.from_pretrained("gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model with specified hidden size
    # Scale other dimensions proportionally
    config = BitNetConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_layers=4,
        num_heads=4,
        num_kv_heads=4,
        ffn_hidden_size=hidden_size * 2,  # Proportional to hidden_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BitNetModel(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters on {device}")
    print()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=cast(int, tokenizer.pad_token_id))

    batch_size = 4
    seq_len = 32

    dataloader = WikiTextDataLoader(
        tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=num_steps,
    )

    # Use standard BitNet settings with corrected WD=0.0 in Stage 2
    lr_scheduler = TwoStageLRScheduler(optimizer, config, num_steps)
    wd_scheduler = TwoStageWDScheduler(optimizer, num_steps)

    model.train()
    step = 0

    with open(output_file, 'w') as f:
        f.write(f"Model Scaling Test: hidden_size={hidden_size}, steps={num_steps}\n")
        f.write("Step,Loss,UniquePredictions,LR,WD\n")

        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            logits = model(batch)
            flat_logits = logits[:, :-1, :].reshape(-1, model.config.vocab_size)
            flat_targets = batch[:, 1:].reshape(-1)

            loss = loss_fn(flat_logits, flat_targets)
            loss.backward()

            # Standard gradient clipping
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            wd_scheduler.step()

            step += 1

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
