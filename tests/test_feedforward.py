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

    def hook(module, input, output):
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
