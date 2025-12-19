"""
Stage 1 training with label smoothing to prevent mode collapse.

Hypothesis: CrossEntropyLoss doesn't penalize vocabulary collapse.
Model learns to predict frequent tokens (lower loss) even though it's bad for generation.

Solution: Label smoothing distributes targets across vocabulary, forcing model
to learn diverse representations rather than collapsing to mode tokens.
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


class LabelSmoothingLoss(nn.Module):
    """CrossEntropyLoss with label smoothing to prevent mode collapse.

    Label smoothing distributes the target probability across all tokens,
    not just the correct one. This encourages the model to learn about
    the full vocabulary rather than collapsing to frequent tokens.

    Args:
        vocab_size: Size of vocabulary
        smoothing: Smoothing parameter (0.0 = standard CE, 0.1 = typical value)
        ignore_index: Index to ignore in loss computation
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed loss.

        Args:
            logits: [batch_size * seq_len, vocab_size]
            targets: [batch_size * seq_len]

        Returns:
            Loss scalar
        """
        # Get log probabilities
        log_probs = self.log_softmax(logits)

        # Create label smoothing distribution
        # Instead of one-hot targets, distribute probability
        if self.smoothing > 0:
            # Uniform distribution over all tokens
            smooth_labels = torch.ones_like(log_probs) / self.vocab_size * self.smoothing

            # Add concentrated probability on correct token
            smooth_labels.scatter_(1, targets.unsqueeze(1), (1.0 - self.smoothing))

            # Compute cross-entropy with smoothed labels
            loss = -(smooth_labels * log_probs).sum(dim=-1)
        else:
            # Standard cross-entropy if no smoothing
            loss = nn.functional.cross_entropy(logits, targets, reduction='none')

        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss


def main():
    """Train with label smoothing to prevent mode collapse."""

    # Parse command line arguments
    num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    smoothing = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    output_file = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/label_smooth_{num_steps}.txt"

    print(f"Stage 1 Training with Label Smoothing")
    print(f"  Total steps: {num_steps}")
    print(f"  Label smoothing: {smoothing}")
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

    # Setup training
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
    )

    # Use label-smoothed loss instead of standard CE
    loss_fn = LabelSmoothingLoss(
        vocab_size=config.vocab_size,
        smoothing=smoothing,
        ignore_index=cast(int, tokenizer.pad_token_id)
    )

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
    print(f"  LR: 0.0015 (standard)")
    print(f"  WD: 0.1 (Stage 1)")
    print(f"  Loss: CrossEntropyLoss with label smoothing ({smoothing})")
    print()

    lr_scheduler = TwoStageLRScheduler(optimizer, config, num_steps)
    wd_scheduler = TwoStageWDScheduler(optimizer, num_steps)
    wd_scheduler.stage1_steps = num_steps + 1  # Keep at 0.1 throughout

    _ = model.train()
    step = 0

    with open(output_file, 'w') as f:
        f.write(f"Label Smoothing Training (smoothing={smoothing}) - {num_steps} steps\n")
        f.write("Step,Loss,UniquePredictions,LR,WD\n")

        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(batch)

            # Shift targets: predict next token given current tokens
            flat_logits = logits[:, :-1, :].reshape(-1, model.config.vocab_size)
            flat_targets = batch[:, 1:].reshape(-1)

            loss = loss_fn(flat_logits, flat_targets)

            # Backward
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update schedulers
            lr_scheduler.step()
            wd_scheduler.step()

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
