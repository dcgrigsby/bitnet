from typing import cast

import torch
import torch.nn as nn
import torch.optim as optim

from bitnet.config import BitNetConfig
from bitnet.train import (
    TwoStageLRScheduler,
    TwoStageWDScheduler,
    train_epoch,
    train_step,
)
from bitnet.transformer import BitNetModel, TransformerBlock
from tests.fixtures import create_dummy_dataloader


def test_train_step_basic():
    """Test that training step works without schedulers"""

    config = BitNetConfig(num_layers=2)
    model = BitNetModel(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    batch = torch.randint(0, config.vocab_size, (2, 10))
    loss = train_step(model, batch, optimizer, loss_fn)

    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.isnan(torch.tensor(loss))


def test_two_stage_lr_scheduler():
    """Test two-stage learning rate scheduler."""

    config = BitNetConfig(num_layers=2, warmup_steps=20)
    model = BitNetModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    total_steps = 100
    lr_scheduler = TwoStageLRScheduler(optimizer, config, total_steps)

    lrs_stage1: list[float] = []
    lrs_stage2: list[float] = []

    for step in range(total_steps):
        batch = torch.randint(0, config.vocab_size, (2, 10))
        _ = train_step(model, batch, optimizer, loss_fn, lr_scheduler=lr_scheduler)

        current_lr = optimizer.param_groups[0]["lr"]
        if step < total_steps // 2:
            lrs_stage1.append(current_lr)
            # Verify stage 1 behavior
            assert len(lrs_stage1) > 0
            assert lrs_stage1[-1] > 0  # Should be positive LR
        else:
            lrs_stage2.append(current_lr)
            # Verify stage 2 has lower LRs than peak in stage 1
            assert len(lrs_stage2) > 0
            assert lrs_stage2[0] < max(lrs_stage1)


def test_two_stage_wd_scheduler():
    """Test two-stage weight decay scheduler."""

    config = BitNetConfig(num_layers=2)
    model = BitNetModel(config)

    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    total_steps = 10
    wd_scheduler = TwoStageWDScheduler(optimizer, total_steps)

    wds_stage1: list[float] = []
    wds_stage2: list[float] = []

    for step in range(total_steps):
        batch = torch.randint(0, config.vocab_size, (2, 10))
        _ = train_step(model, batch, optimizer, loss_fn, wd_scheduler=wd_scheduler)

        current_wd = optimizer.param_groups[0]["weight_decay"]
        if step < total_steps // 2:
            wds_stage1.append(current_wd)
        else:
            wds_stage2.append(current_wd)

    # Verify stage 1 uses weight decay
    assert len(wds_stage1) > 0
    assert wds_stage1[0] == 0.1

    # Verify stage 2 disables weight decay
    assert len(wds_stage2) > 0
    assert all(wd == 0.0 for wd in wds_stage2)


def test_gradient_flow_with_quantization():
    """Test that gradients flow through quantized layers."""

    config = BitNetConfig(num_layers=2)
    model = BitNetModel(config)

    batch = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(batch)
    loss = logits.sum()
    loss.backward()

    # Check that gradients exist in all major components
    assert model.token_embeddings.weight.grad is not None
    block = cast(TransformerBlock, model.blocks[0])
    assert block.attention.qkv_proj.weight.grad is not None
    assert block.feedforward.gate_up.weight.grad is not None
    assert model.lm_head.weight.grad is not None

    # Check gradients are bounded (no NaN/Inf)
    for param in model.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()


def test_parameters_update_with_schedulers():
    """Test that parameters update correctly with two-stage schedulers."""

    config = BitNetConfig(num_layers=2)
    model = BitNetModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Store initial parameters
    initial_params = [p.clone() for p in model.parameters()]

    total_steps = 50
    lr_scheduler = TwoStageLRScheduler(optimizer, config, total_steps)
    wd_scheduler = TwoStageWDScheduler(optimizer, total_steps)

    # Training steps with schedulers
    for _ in range(total_steps):
        batch = torch.randint(0, config.vocab_size, (2, 10))
        _ = train_step(
            model,
            batch,
            optimizer,
            loss_fn,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
        )

    # Check that some parameters have changed
    params_changed = False
    for p_init, p_now in zip(initial_params, model.parameters()):
        if not torch.allclose(p_init, p_now):
            params_changed = True
            break

    assert params_changed, "No parameters updated during training"


def test_train_epoch_with_schedulers():
    """Test full epoch training with two-stage schedulers."""

    config = BitNetConfig(num_layers=2)
    model = BitNetModel(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    dataloader = create_dummy_dataloader(
        config, num_batches=8, batch_size=2, seq_len=16
    )

    total_steps = 8  # One epoch
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

    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.isnan(torch.tensor(loss))


def test_higher_bitnet_learning_rate():
    """Test that BitNet uses higher learning rates than standard FP16.

    BitNet b1.58 requires 5-7x higher learning rates than full-precision
    models for stable training (paper finding).
    """

    config = BitNetConfig(num_layers=2)

    # BitNet learning rate
    bitnet_lr = config.learning_rate  # Default: 1.5e-3

    # Standard FP16 learning rate would be ~2e-4
    fp16_lr = 2e-4

    # Verify BitNet LR is significantly higher
    assert bitnet_lr > fp16_lr * 3  # At least 3x higher
    assert bitnet_lr >= 1.0e-3  # In the 1.5e-3 range
