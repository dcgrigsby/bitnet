import sys
from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, PreTrainedTokenizer

from bitnet.config import BitNetConfig
from bitnet.data import WikiTextDataLoader
from bitnet.train import (
    TwoStageLRScheduler,
    TwoStageWDScheduler,
    train_step,
)
from bitnet.transformer import BitNetModel


def main():
    """Simple training example with WikiText-2 and two-stage schedulers."""

    # Parse command line arguments
    num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 48

    # Load tokenizer
    tokenizer = cast(PreTrainedTokenizer, GPT2Tokenizer.from_pretrained("gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    config = BitNetConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=192,
        num_layers=3,
        num_heads=3,
        num_kv_heads=3,
        ffn_hidden_size=384,
        stage2_weight_decay=0.05,  # Reduced from 0.0 to prevent mode collapse
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Test without quantization to diagnose if quantization is the issue
    model = BitNetModel(config, disable_quant=True).to(device)

    print(
        f"Model: {sum(p.numel() for p in model.parameters()):,} parameters on {device}"
    )

    # Setup training
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
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

    # Training loop with two-stage schedulers
    print(f"Training for {num_steps} steps...")
    print(f"  Stage 1: steps 0-{num_steps // 2} (higher LR, WD={config.weight_decay})")
    print(f"  Stage 2: steps {num_steps // 2 + 1}-{num_steps} (lower LR, WD={config.stage2_weight_decay})")
    print()

    lr_scheduler = TwoStageLRScheduler(optimizer, config, num_steps)
    wd_scheduler = TwoStageWDScheduler(
        optimizer,
        num_steps,
        stage1_wd=config.weight_decay,
        stage2_wd=config.stage2_weight_decay,
    )

    _ = model.train()
    step = 0

    # Diagnostic: capture initial weights
    initial_weights = {name: param.clone().detach() for name, param in model.named_parameters()}

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
        if step % 16 == 0:
            current_lr = cast(float, optimizer.param_groups[0]["lr"])
            current_wd = cast(float, optimizer.param_groups[0]["weight_decay"])
            stage = "Stage 2" if step > num_steps // 2 else "Stage 1"
            print(
                f"Step {step}/{num_steps} ({stage}): Loss = {loss:.4f}, LR = {current_lr:.6f}, WD = {current_wd:.1f}"
            )

    # Diagnostic: check if weights changed
    print("\nWeight change diagnostic:")
    total_weight_change = 0.0
    for name, param in model.named_parameters():
        if name in initial_weights:
            change = torch.abs(param - initial_weights[name]).mean().item()
            total_weight_change += change
            if change < 1e-7:
                print(f"  WARNING: {name} barely changed ({change:.2e})")
    print(f"  Average weight change: {total_weight_change / len(initial_weights):.6f}")

    # Test inference
    print("\nInference examples:")
    _ = model.eval()
    test_sentences = [
        "The quick brown fox",
        "Machine learning is",
        "Hello world",
    ]

    with torch.no_grad():
        for test_input in test_sentences:
            input_ids = cast(torch.Tensor, tokenizer.encode(test_input, return_tensors="pt")).to(device)
            logits = model(input_ids)

            # Diagnostic: show logit statistics
            last_logits = logits[0, -1, :]  # Last token's logits
            top_k_values, top_k_indices = torch.topk(last_logits, k=5)
            top_k_tokens = [tokenizer.decode([idx.item()]) for idx in top_k_indices]
            entropy = -torch.sum(torch.softmax(last_logits, dim=0) * torch.log_softmax(last_logits, dim=0))

            predictions = torch.argmax(logits, dim=-1)
            predicted_text = cast(str, tokenizer.decode(predictions[0], skip_special_tokens=True))
            print(f"  Input:  '{test_input}'")
            print(f"    Output: '{predicted_text}'")
            print(f"    Top 5 next tokens: {list(zip(top_k_tokens, [f'{v:.2f}' for v in top_k_values]))}")
            print(f"    Entropy: {entropy:.4f}")


if __name__ == "__main__":
    main()
