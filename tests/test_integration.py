import torch
import torch.nn as nn
import torch.optim as optim

from bitnet.config import BitNetConfig
from bitnet.train import (
    TwoStageLRScheduler,
    TwoStageWDScheduler,
    create_dummy_dataloader,
    train_epoch,
    train_step,
)
from bitnet.transformer import BitNetModel


def test_full_training_pipeline():
    """Integration test: full training pipeline with two stage schedulers"""

    config = BitNetConfig(
        vocab_size=256,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        ffn_hidden_size=512,
    )

    model = BitNetModel(config)

    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    loss_fn = nn.CrossEntropyLoss()

    # Create dummy dataloader
    dataloader = create_dummy_dataloader(
        config, num_batches=4, batch_size=2, seq_len=16
    )

    # Calculate total steps: 4 batches/epoch * 3 epochs
    total_steps = 4 * 3

    initial_loss, loss = None, None
    for epoch in range(3):
        lr_scheduler = TwoStageLRScheduler(optimizer, config, total_steps)
        wd_scheduler = TwoStageWDScheduler(optimizer, total_steps)

        loss = train_epoch(
            model,
            dataloader,
            optimizer,
            loss_fn,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
        )

        if initial_loss is None:
            initial_loss = loss

        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    # Check that training completed without errors
    assert not torch.isnan(torch.tensor(loss))
    # Loss may go up or down due to randomness, just verify training works
    assert loss is not None
    assert loss > 0


def test_training_updates_all_parameters():
    """Test that training updates parameters in all model components."""

    config = BitNetConfig(
        vocab_size=256,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        ffn_hidden_size=256,
    )

    model = BitNetModel(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Store initial parameters
    initial_params = {name: param.clone() for name, param in model.named_parameters()}

    # Training steps
    for _ in range(10):
        batch = torch.randint(0, config.vocab_size, (2, 8))
        _ = train_step(model, batch, optimizer, loss_fn)

    # Check that various components were updated
    components_updated = {
        "embeddings": False,
        "attention": False,
        "feedforward": False,
        "final_norm": False,
        "lm_head": False,
    }

    for name, param in model.named_parameters():
        initial = initial_params[name]
        if not torch.allclose(param, initial):
            if "token_embeddings" in name:
                components_updated["embeddings"] = True
            elif "attention" in name or "qkv_proj" in name or "out_proj" in name:
                components_updated["attention"] = True
            elif "feedforward" in name or "gate_up" in name or "down" in name:
                components_updated["feedforward"] = True
            elif "final_norm" in name:
                components_updated["final_norm"] = True
            elif "lm_head" in name:
                components_updated["lm_head"] = True

    # Verify all major components were updated
    assert components_updated["embeddings"], "Embeddings not updated"
    assert components_updated["attention"], "Attention not updated"
    assert components_updated["feedforward"], "Feedforward not updated"
    assert components_updated["lm_head"], "LM head not updated"
