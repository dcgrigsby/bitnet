"""Analyze training dynamics to identify mode collapse."""
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer
from typing import cast

from bitnet.config import BitNetConfig
from bitnet.data import WikiTextDataLoader
from bitnet.train import TwoStageLRScheduler, TwoStageWDScheduler, train_step
from bitnet.transformer import BitNetModel


def analyze_per_token_logit_distribution(model, batch_size=16, seq_len=32):
    """Analyze how logits are distributed across the vocabulary."""
    tokenizer = cast(GPT2Tokenizer, GPT2Tokenizer.from_pretrained("gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = WikiTextDataLoader(
        tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=1,  # Just one batch
    )

    model.eval()
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch)  # Shape: [batch_size, seq_len, vocab_size]
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)  # Shape: [batch_size, seq_len, vocab_size]

    # Flatten to [batch_size*seq_len, vocab_size]
    all_logits_flat = all_logits.reshape(-1, all_logits.shape[-1])

    # Compute softmax to get probabilities
    probs = torch.softmax(all_logits_flat, dim=-1)

    # Get top-k probabilities for each position
    top_k = 5
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

    # Analyze concentration: what fraction of probability mass is in top-1 token?
    top1_probs = probs.max(dim=-1).values
    concentration = top1_probs.mean().item()

    entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()

    print(f"  Top-1 concentration: {concentration:.3f} (1.0=degenerate, {1/50257:.4f}=uniform)")
    print(f"  Entropy: {entropy:.2f} (max={torch.log(torch.tensor(50257.0)).item():.2f} for uniform)")

    # Find most common predicted tokens
    top_pred_tokens = top_indices[:, 0]  # Top-1 predictions for all positions
    unique, counts = torch.unique(top_pred_tokens, return_counts=True)
    sorted_counts, sorted_indices = torch.sort(counts, descending=True)

    print(f"  Top 5 most-predicted tokens (out of {len(unique)} unique):")
    for i in range(min(5, len(unique))):
        token_id = unique[sorted_indices[i]].item()
        count = sorted_counts[i].item()
        percentage = 100 * count / len(top_pred_tokens)
        token_text = tokenizer.decode([token_id])
        print(f"    {i+1}. Token {token_id} ('{token_text}'): {count} times ({percentage:.1f}%)")

    return concentration, entropy


def analyze_weight_statistics(model):
    """Analyze weight statistics to detect mode collapse."""
    print("Weight statistics:")

    total_params = 0
    zero_or_near_zero = 0
    large_weights = 0

    for name, param in model.named_parameters():
        if "weight" in name:
            total_params += param.numel()

            # Count near-zero weights (bitnet should have ternary {-1, 0, 1})
            near_zero = (param.abs() < 0.1).sum().item()
            zero_or_near_zero += near_zero

            # Count weights outside [-1, 1] (shouldn't happen with ternary quant)
            large = (param.abs() > 1.1).sum().item()
            large_weights += large

    zero_pct = 100 * zero_or_near_zero / total_params
    print(f"  Zero or near-zero weights: {zero_pct:.1f}%")

    if large_weights > 0:
        print(f"  ⚠ Found {large_weights} weights outside [-1, 1] (unexpected for ternary)")
    else:
        print(f"  ✓ All weights in expected range")

    return zero_pct


def compare_token_frequency(model, tokenizer, text_sample):
    """Compare predicted tokens vs ground truth token frequency."""
    model.eval()

    input_ids = tokenizer.encode(text_sample, return_tensors="pt")

    with torch.no_grad():
        logits = model(input_ids)

    # Get predicted tokens
    probs = torch.softmax(logits[0, :, :], dim=-1)
    predicted = torch.argmax(probs, dim=-1)

    # Count frequency of predicted tokens
    unique, counts = torch.unique(predicted, return_counts=True)
    sorted_counts, sorted_indices = torch.sort(counts, descending=True)

    print(f"Predicted token frequency in sample:")
    for i in range(min(5, len(unique))):
        token_id = unique[sorted_indices[i]].item()
        count = sorted_counts[i].item()
        token_text = tokenizer.decode([token_id])
        print(f"  {i+1}. '{token_text}': {count} times")


if __name__ == "__main__":
    # Load tokenizer and config
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

    # Train for a small number of steps to analyze dynamics
    print("Training model to analyze dynamics...")
    print("="*50)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=cast(int, tokenizer.pad_token_id))

    dataloader = WikiTextDataLoader(
        tokenizer,
        batch_size=4,
        seq_len=32,
        num_steps=20,
    )

    model.train()
    num_steps = 20

    lr_scheduler = TwoStageLRScheduler(optimizer, config, num_steps)
    wd_scheduler = TwoStageWDScheduler(optimizer, num_steps)

    for step, batch in enumerate(dataloader, 1):
        batch = batch.to(device)
        loss = train_step(
            model, batch, optimizer, loss_fn,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
        )

        if step % 5 == 0:
            print(f"Step {step}/{num_steps}: Loss = {loss:.4f}")

    print("\n" + "="*50)
    print("\nAnalyzing post-training dynamics:")
    print("="*50)

    concentration, entropy = analyze_per_token_logit_distribution(model)
    print()

    weight_zero_pct = analyze_weight_statistics(model)
    print()

    print("Comparing to ground truth:")
    compare_token_frequency(model, tokenizer, "The quick brown fox jumps over the lazy dog")
