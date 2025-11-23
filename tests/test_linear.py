import pytest
import torch

from bitnet.linear import BitLinear
from bitnet.quant import activation_quant, weight_quant


def test_bitlinear_forward_backward():
    """Test BitLinear supports forward and backward passes."""

    layer = BitLinear(768, 512)
    x = torch.randn(2, 10, 768, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    # Check gradients exist
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape


def test_bitlinear_weight_quantization():
    """Test that weights are quantized to ternary values during forward pass"""

    layer = BitLinear(100, 50)
    x = torch.randn(2, 10, 100)

    # Capture effective weights used in computation
    original_weight = layer.weight.clone()

    # Run forward pass
    y = layer(x)

    # Manually quantize weights
    w_quant = weight_quant(original_weight)

    # Quantized weights should have 2 or 3 unique values
    unique_vals = torch.unique(w_quant)
    assert 2 <= len(unique_vals) <= 3

    # Original weights should remain unchanged (not in-place)
    assert torch.allclose(layer.weight, original_weight)
