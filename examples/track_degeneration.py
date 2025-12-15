"""Track when and how degenerate decoding emerges during training."""
import sys
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from typing import cast

from bitnet.config import BitNetConfig
from bitnet.data import WikiTextDataLoader
from bitnet.train import TwoStageLRScheduler, TwoStageWDScheduler, train_step
from bitnet.transformer import BitNetModel


def compute_token_concentration(model):
    """Compute top-1 probability concentration (1.0 = degenerate, 0 = uniform)."""
    tokenizer = cast(GPT2Tokenizer, GPT2Tokenizer.from_pretrained("gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Sample from one batch
    dataloader = WikiTextDataLoader(
        tokenizer, batch_size=4, seq_len=32, num_steps=1
    )

    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            top1_probs = probs.max(dim=-1).values
            concentration = top1_probs.mean().item()
            break

    return concentration


def compute_entropy(model):
    """Compute entropy of predicted distribution."""
    tokenizer = cast(GPT2Tokenizer, GPT2Tokenizer.from_pretrained("gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    dataloader = WikiTextDataLoader(
        tokenizer, batch_size=4, seq_len=32, num_steps=1
    )

    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            break

    return entropy


def count_unique_predictions(model):
    """Count how many unique tokens the model predicts."""
    tokenizer = cast(GPT2Tokenizer, GPT2Tokenizer.from_pretrained("gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    dataloader = WikiTextDataLoader(
        tokenizer, batch_size=4, seq_len=32, num_steps=1
    )

    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            unique_count = torch.unique(predictions).numel()
            break

    return unique_count


if __name__ == "__main__":
    # Parse arguments
    num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    # Setup
    tokenizer = cast(GPT2Tokenizer, GPT2Tokenizer.from_pretrained("gpt2"))
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

    print(
        f"Model: {sum(p.numel() for p in model.parameters()):,} parameters on {device}"
    )
    print(f"Training for {num_steps} steps...")
    print(f"Tracking: Loss | Top-1 Conc | Entropy | Unique Tokens")
    print("="*60)

    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=cast(int, tokenizer.pad_token_id))

    dataloader = WikiTextDataLoader(
        tokenizer, batch_size=4, seq_len=32, num_steps=num_steps
    )

    model.train()
    lr_scheduler = TwoStageLRScheduler(optimizer, config, num_steps)
    wd_scheduler = TwoStageWDScheduler(optimizer, num_steps)

    step = 0
    for batch in dataloader:
        batch = batch.to(device)
        loss = train_step(
            model, batch, optimizer, loss_fn,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
        )
        step += 1

        # Evaluate every 5 steps
        if step % 5 == 0:
            concentration = compute_token_concentration(model)
            entropy = compute_entropy(model)
            unique = count_unique_predictions(model)

            stage = "Stage 1" if step <= num_steps // 2 else "Stage 2"

            print(
                f"Step {step:3d} ({stage}): "
                f"Loss={loss:6.2f} | "
                f"Conc={concentration:.4f} | "
                f"Ent={entropy:5.2f} | "
                f"Unique={unique:4d}"
            )

    print("="*60)
