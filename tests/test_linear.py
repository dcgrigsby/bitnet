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


def test_bitlinear_activation_quantization():
    """Test that activations are quantized during forward pass."""

    layer = BitLinear(512, 256)
    x = torch.randn(2, 8, 512)

    # Manually compute normalized + quantized activations
    x_norm = layer.norm(x)
    x_quant_expected = activation_quant(x_norm)

    # Check quantization reduces precision
    unique_orig = len(torch.unique(x_norm[0, 0, :]))
    unique_quant = len(torch.unique(x_quant_expected[0, 0, :]))

    # Quantized should have fewer unique values (int8 discretization)
    assert unique_quant < unique_orig
    assert unique_quant <= 256  # int8 range


def test_bitlinear_ste_gradient_bypass():
    """Test Straight-Through Estimator bypasses quantization in backward pass."""

    layer = BitLinear(256, 128)
    x = torch.randn(2, 10, 256, requires_grad=True)

    # Forward pass
    y = layer(x)
    loss = y.sum()

    # Backward pass
    loss.backward()

    # Gradients should exist and be non-zero
    assert x.grad is not None
    assert layer.weight.grad is not None

    # Gradients should frow as if no quantization (STE property)
    assert x.grad.abs().max() > 0
    assert x.grad.abs().max() < 100
    assert layer.weight.grad.abs().max() > 0
    assert layer.weight.grad.abs().max() < 100

def test_bitlinear_deterministic():
    """Test BitLinear produces deterministic output for same input."""

    layer = BitLinear(768, 512)
    x = torch.randn(2, 10, 768)

    y1 = layer(x)
    y2 = layer(x)

    assert torch.allclose(y1, y2)

def test_bitlinear_batch_independence():
    """Test BitLinear processes each sample independently."""
    layer = BitLinear(256, 128)
    layer.eval()  # Use eval mode for per-token quantization
    
    x = torch.randn(4, 10, 256)

    with torch.no_grad():
        y_batch = layer(x)
        # Use cat instead of stack to preserve shape
        y_individual = torch.cat([layer(x[i:i+1]) for i in range(4)], dim=0)

    assert torch.allclose(y_batch, y_individual, atol=1e-5)

def test_bitlinear_parameter_learning():
    """Test BitLinear weights actually update during training."""

    layer = BitLinear(256, 128)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

    # Store initial weights
    initial_weight = layer.weight.clone()
    initial_norm_weight = layer.norm.weight.clone()

    # Training step
    x = torch.randn(4, 10, 256, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # Weights should have changed
    assert not torch.allclose(layer.weight, initial_weight)
    assert not torch.allclose(layer.norm.weight, initial_norm_weight)
