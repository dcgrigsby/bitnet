"""
Stage 1 training with aggressive gradient clipping.

Hypothesis: Reduced LR worked because it slows weight changes.
Maybe aggressive gradient clipping has the same effect - prevents weights from diverging too fast.
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
    """Train with aggressive gradient clipping."""

    num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    clip_norm = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1  # vs default 1.0
    output_file = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/grad_clip_{clip_norm}_{num_steps}.txt"

    print(f"Stage 1 Training with Aggressive Gradient Clipping")
    print(f"  Total steps: {num_steps}")
    print(f"  Gradient clip norm: {clip_norm} (vs default 1.0)")
    print(f"  Output file: {output_file}")
    print()

    tokenizer = cast(PreTrainedTokenizer, GPT2Tokenizer.from_pretrained("gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    print(f"Training config:")
    print(f"  LR: 0.0015 (standard)")
    print(f"  WD: 0.1 (Stage 1)")
    print(f"  Gradient clip norm: {clip_norm}")
    print()

    lr_scheduler = TwoStageLRScheduler(optimizer, config, num_steps)
    wd_scheduler = TwoStageWDScheduler(optimizer, num_steps)
    wd_scheduler.stage1_steps = num_steps + 1

    model.train()
    step = 0

    with open(output_file, 'w') as f:
        f.write(f"Aggressive Gradient Clipping (clip={clip_norm}) - {num_steps} steps\n")
        f.write("Step,Loss,UniquePredictions,LR,WD\n")

        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            logits = model(batch)
            flat_logits = logits[:, :-1, :].reshape(-1, model.config.vocab_size)
            flat_targets = batch[:, 1:].reshape(-1)

            loss = loss_fn(flat_logits, flat_targets)
            loss.backward()

            # Aggressive gradient clipping
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

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
