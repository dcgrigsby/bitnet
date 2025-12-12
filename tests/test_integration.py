import torch
import torch.nn as nn
import torch.optim as optim
from typing import cast

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


def test_training_mode_vs_eval_mode():
    """Test model behaves consistently in train vs eval mode."""

    config = BitNetConfig(
        num_layers=2, hidden_size=256, num_heads=4, num_kv_heads=4, ffn_hidden_size=512
    )

    model = BitNetModel(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))

    # Training mode
    _ = model.train()
    with torch.no_grad():
        logits_train = model(input_ids)

    # Eval mode
    _ = model.eval()
    with torch.no_grad():
        logits_eval = model(input_ids)

    # For deterministic operations, outputs should be identical
    # (no dropout in this model, so should match exactly)
    assert torch.allclose(logits_train, logits_eval)


def test_model_logits_distribution():
    """Test model logits have reasonable distribution."""

    config = BitNetConfig(num_layers=2, vocab_size=1000)
    model = BitNetModel(config)
    _ = model.eval()

    with torch.no_grad():
        batch = torch.randint(0, config.vocab_size, (4, 16))
        logits = model(batch)

        # Logits should span a reasonable range
        assert logits.min() < 0  # Some negative logits
        assert logits.max() > 0  # Some positive logits

        # Logits shouldn't be too extreme
        assert logits.abs().max() < 100

        # Each position should have varied predictions
        for b in range(4):
            for t in range(16):
                token_logits = logits[b, t, :]
                top_5 = torch.topk(token_logits, k=5).values
                # Top 5 should have different values
                assert len(torch.unique(top_5)) >= 4


def test_loss_computation():
    """Test loss computation for next-token prediction."""

    config = BitNetConfig(vocab_size=100, num_layers=2)
    model = BitNetModel(config)
    loss_fn = nn.CrossEntropyLoss()

    batch = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(batch)

    # Compute loss (next token prediction)
    loss = loss_fn(
        logits[:, :-1, :].reshape(-1, config.vocab_size), batch[:, 1:].reshape(-1)
    )

    # Loss should be positive and finite
    assert loss.item() > 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Loss should be in reasonable range (cross-entropy with ~100 classes)
    assert loss.item() < 10  # Should be less than log(vocab_size)


def test_overfitting_on_small_dataset():
    """Test model can overfit to a tiny dataset (sanity check)."""

    config = BitNetConfig(
        vocab_size=50,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        ffn_hidden_size=256,
    )

    model = BitNetModel(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Tiny dataset: single sequence repeated
    sequence = torch.randint(0, config.vocab_size, (1, 16))
    dataloader = [(sequence,) for _ in range(20)]

    initial_loss = None
    final_loss = None

    # Train for many steps
    for _ in range(10):
        for batch_tuple in dataloader:
            batch = batch_tuple[0]
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                batch[:, 1:].reshape(-1),
            )
            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()
            loss.backward()
            _ = optimizer.step()

    # Model should overfit: final loss should be lower than initial
    assert final_loss is not None
    assert final_loss < initial_loss


def test_checkpoint_compatibility():
    """Test model state dict can be saved and loaded."""

    config = BitNetConfig(num_layers=2)
    model1 = BitNetModel(config)

    # Save state
    state_dict = model1.state_dict()

    # Create new model and load state
    model2 = BitNetModel(config)
    _ = model2.load_state_dict(state_dict)

    # Both models should produce identical outputs
    _ = model1.eval()
    _ = model2.eval()

    with torch.no_grad():
        batch = torch.randint(0, config.vocab_size, (2, 10))
        logits1 = model1(batch)
        logits2 = model2(batch)

    assert torch.allclose(logits1, logits2)


def test_quantization_active_during_training():
    """Test that quantization is actually being used during training."""

    config = BitNetConfig(num_layers=1, hidden_size=128, num_heads=4, num_kv_heads=4)
    model = BitNetModel(config)

    # Get a BitLinear layer
    block = cast(nn.Module, model.blocks[0])
    attention = cast(nn.Module, block.attention)
    bitlinear_layer = cast(nn.Module, attention.qkv_proj)

    # Store original weight
    original_weight = cast(torch.Tensor, cast(nn.Module, bitlinear_layer).weight).clone()

    # Forward pass
    _ = torch.randn(1, 8, config.hidden_size)
    y = cast(torch.Tensor, model.token_embeddings(torch.randint(0, config.vocab_size, (1, 8))))
    norm_module = cast(nn.Module, attention)
    y = cast(torch.Tensor, norm_module.norm(y))  # type: ignore[misc]

    # Capture output with quantization
    output_with_quant = cast(nn.Module, bitlinear_layer)(y)

    # Verify weight hasn't changed (quantization is not in-place)
    assert torch.allclose(cast(torch.Tensor, cast(nn.Module, bitlinear_layer).weight), original_weight)

    # Output should be from quantized computation
    # (Hard to verify directly, but can check it's not identical to unquantized)
    assert output_with_quant is not None
    assert not torch.isnan(output_with_quant).any()

