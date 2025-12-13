import sys

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer

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
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
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
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

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
    print(f"  Stage 1: steps 0-{num_steps // 2} (higher LR, WD=0.1)")
    print(f"  Stage 2: steps {num_steps // 2 + 1}-{num_steps} (lower LR, WD=0.0)")
    print()

    lr_scheduler = TwoStageLRScheduler(optimizer, config, num_steps)
    wd_scheduler = TwoStageWDScheduler(optimizer, num_steps)

    _ = model.train()
    step = 0

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
            current_lr = optimizer.param_groups[0]["lr"]
            current_wd = optimizer.param_groups[0]["weight_decay"]
            stage = "Stage 2" if step > num_steps // 2 else "Stage 1"
            print(
                f"Step {step}/{num_steps} ({stage}): Loss = {loss:.4f}, LR = {current_lr:.6f}, WD = {current_wd:.1f}"
            )

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
            input_ids = tokenizer.encode(test_input, return_tensors="pt").to(device)
            logits = model(input_ids)
            predictions = torch.argmax(logits, dim=-1)
            predicted_text = tokenizer.decode(predictions[0], skip_special_tokens=True)
            print(f"  Input:  '{test_input}' -> '{predicted_text}'")


if __name__ == "__main__":
    main()
