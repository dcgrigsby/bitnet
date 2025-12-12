import pytest
import torch
import torch.nn.functional as F

from bitnet.feedforward import FeedForward


def test_feedforward_applies_normalization():
    """Test FFN applies normalization before projections."""
    ffn = FeedForward(768, 3072)
    x = torch.randn(2, 10, 768) * 10  # Varying magnitude

    # Hook to capture normalized input
    normalized_input = []

    def hook(_module, _input, output):
        normalized_input.append(output)

    handle = ffn.norm.register_forward_hook(hook)

    y = ffn(x)

    handle.remove()

    # Verify normalization was applied
    assert len(normalized_input) > 0
    x_norm = normalized_input[0]

    # RMS should be approximately 1.0 per token
    rms = torch.sqrt((x_norm**2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


def test_feedfoward_swiglu_activation():
    """Test FFN uses SwiGLU (squared ReLU gating)."""

    ffn = FeedForward(256, 512)
    x = torch.randn(2, 8, 256)

    # Manually compute what SwiGLU should produce
    x_norm = ffn.norm(x)
    gate_up = ffn.gate_up(x_norm)
    gate, up = gate_up.chunk(2, dim=-1)

    # SwiGLU uses squared ReLU for gate
    gate_activated = F.relu(gate) ** 2

    # Check gate activation produces non-negative values
    assert (gate_activated >= 0).all()

    # Check gate has been squared (values should be larger or smaller)
    gate_relu = F.relu(gate)
    # For positive values, squared should differ from original
    positive_mask = gate_relu > 0.1
    if positive_mask.any():
        # Squared values should differe from ReLU values
        assert not torch.allclose(
            gate_relu[positive_mask], gate_activated[positive_mask]
        )


def test_feedforward_gating_modulates_output():
    """Test that forward() properly applies gating to modulate output."""

    ffn = FeedForward(256, 512)
    x = torch.randn(2, 8, 256)
    output = ffn(x)

    # Manually compute expected output
    with torch.no_grad():
        x_norm = ffn.norm(x)
        gate_up = ffn.gate_up(x_norm)
        gate, up = gate_up.chunk(2, dim=-1)
        gate_activated = F.relu(gate) ** 2
        hidden = gate_activated * up
        expected_output = ffn.down(hidden)

    # Verify forward() matches expected computation
    assert torch.allclose(output, expected_output, atol=1e-5)

    # Verify gating zeros out features where gate is near-zero
    zero_gate_mask = gate_activated.abs() < 1e-6
    if zero_gate_mask.any():
        assert torch.allclose(
            hidden[zero_gate_mask], torch.zeros_like(hidden[zero_gate_mask]), atol=1e-5
        )


def test_feedforward_forward_backward():
    """Test FFN suppost forward and backward with proper gradient flow."""

    hidden_size = 768
    ffn_hidden_size = 3072

    ffn = FeedForward(hidden_size, ffn_hidden_size)
    x = torch.randn(2, 10, hidden_size, requires_grad=True)
    y = ffn(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert ffn.gate_up.weight.grad is not None
    assert ffn.down.weight.grad is not None
    assert ffn.norm.weight.grad is not None
    assert not torch.isnan(y).any()

    # Gradients should be bounded (important for BitNet stability)
    assert ffn.gate_up.weight.grad.abs().max() < 1000
    assert ffn.down.weight.grad.abs().max() < 1000


def test_feedforward_parameter_learning():
    """Test FFN parameters update during training."""
    ffn = FeedForward(256, 512)
    optimizer = torch.optim.SGD(ffn.parameters(), lr=0.1)

    # Store initial weights
    initial_gate_up = ffn.gate_up.weight.clone()
    initial_down = ffn.down.weight.clone()

    # Training step
    x = torch.randn(4, 10, 256, requires_grad=True)
    y = ffn(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # Weights should have changed
    assert not torch.allclose(ffn.gate_up.weight, initial_gate_up)
    assert not torch.allclose(ffn.down.weight, initial_down)


def test_feedforward_deterministic():
    """Test FFN produces deterministic output for same input."""
    ffn = FeedForward(768, 3072)
    x = torch.randn(2, 10, 768)

    y1 = ffn(x)
    y2 = ffn(x)

    assert torch.allclose(y1, y2)


def test_feedforward_batch_independence():
    """Test FFN processes each sample independently."""
    ffn = FeedForward(256, 512)
    x = torch.randn(4, 10, 256)

    # Process batch
    y_batch = ffn(x)

    # Process samples individually
    y_individual = torch.cat([ffn(x[i : i + 1]) for i in range(4)], dim=0)

    # Results should match (batch independence)
    assert torch.allclose(y_batch, y_individual, atol=1e-5)


def test_feedforward_expands_then_contracts():
    """Test FFN expands to ffn_hidden_size then contracts back."""
    hidden_size = 768
    ffn_hidden_size = 3072

    ffn = FeedForward(hidden_size, ffn_hidden_size)
    x = torch.randn(2, 10, hidden_size)

    # Hook to capture intermediate hidden state
    hidden_states = []

    def hook(_module, input, _output):
        hidden_states.append(input[0])

    handle = ffn.down.register_forward_hook(hook)
    y = ffn(x)
    handle.remove()

    # Check intermediate size
    assert len(hidden_states) > 0
    hidden = hidden_states[0]
    assert hidden.shape == (2, 10, ffn_hidden_size)
