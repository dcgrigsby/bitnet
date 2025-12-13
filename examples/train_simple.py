import torch
import torch.nn as nn
import torch.optim as optim
# from transformers import LlamaTokenizer
from transformers import GPT2Tokenizer

from bitnet.config import BitNetConfig
from bitnet.train import (
    TwoStageLRScheduler,
    TwoStageWDScheduler,
    create_dummy_dataloader,
    train_epoch,
)
from bitnet.transformer import BitNetModel


def main():
    """Simple training example with GPT-2 tokenizer and two-stage schedulers."""

    # Load tokenizer
    # tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
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

    # Create dummy data
    num_epochs = 3
    num_batches_per_epoch = 16
    batch_size = 4
    seq_len = 32
    total_steps = num_batches_per_epoch * num_epochs

    dataloader = create_dummy_dataloader(
        config,
        num_batches=num_batches_per_epoch,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    # Training loop with two-stage schedulers
    print(f"Training for {num_epochs} epochs ({total_steps} steps)...")
    print(f"  Stage 1: steps 0-{total_steps // 2} (higher LR, WD=0.1)")
    print(f"  Stage 2: steps {total_steps // 2 + 1}-{total_steps} (lower LR, WD=0.0)")
    print()

    step_count = 0
    for epoch in range(num_epochs):
        lr_scheduler = TwoStageLRScheduler(optimizer, config, total_steps)
        wd_scheduler = TwoStageWDScheduler(optimizer, total_steps)

        for _ in range(step_count):
            lr_scheduler.step()
            wd_scheduler.step()

        loss = train_epoch(
            model,
            dataloader,
            optimizer,
            loss_fn,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
        )

        step_count += num_batches_per_epoch
        current_lr = optimizer.param_groups[0]["lr"]
        current_wd = optimizer.param_groups[0]["weight_decay"]
        stage = "Stage 2" if step_count > total_steps // 2 else "Stage 1"
        print(
            f"Epoch {epoch + 1}/{num_epochs} ({stage}): Loss = {loss:.4f}, LR = {current_lr:.6f}, WD = {current_wd:.1f}"
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
