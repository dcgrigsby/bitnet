import pytest
import torch

from bitnet.quant import activation_quant, weight_quant


def test_activation_quant_per_token_scaling():
    """
    Test that each token is independently scaled to int8 range.

    Per-token quantization (not per-tensor) is standard for LLM quantization
    to handle activation outliers. BitNet uses per-token absmax quantization
    for 8-bit activations, which pairs with its unique ternary weight
    quantization. Each token's max absolute value should be preserved.
    """

    x = torch.randn(4, 10, 768)
    y = activation_quant(x)

    for b in range(4):
        for t in range(10):
            original_max = x[b, t, :].abs().max()
            quantized_max = y[b, t, :].abs().max()

            if original_max >= 1e-4:  # Skip near zero tokens
                # After quantization, max should be preserved (withing 2% with rounding)
                assert torch.isclose(quantized_max, original_max, rtol=0.02)


def test_activation_quant_produces_discrete_levels():
    """
    Test that quantization produces discrete, not continous, vales.

    Verifies that activation quantization actually happened and didn't
    just return the input unchanged.
    """

    x = torch.randn(2, 8, 768)
    y = activation_quant(x)

    # For a single token, quantized to int8 means at most 256 unique values
    for b in range(2):
        for t in range(8):
            token_values = y[b, t, :]
            unique_count = len(torch.unique(token_values))

            # Should be quantized to int8 levels
            assert unique_count <= 256


def test_activation_quant_gradient_flow():
    """
    Test that gradients flow through quantization via STE.

    How this works:
    1. PyTorch builds a computational graph: x → activation_quant → y → sum → loss
    2. Each operation stores references to its inputs in the graph
    3. loss.backward() walks the graph backward (loss → y → x)
    4. At each step, PyTorch computes gradients via chain rule
    5. Finally writes the gradient to x.grad (x itself never changes)

    Why this matters:
    If quantization blocks gradients, x.grad will be None or zero, meaning
    model parameters won't receive gradients during training and the model
    can't learn. The STE (Straight-Through Estimator) ensures gradients
    bypass the discrete quantization operation so training works.
    """

    x = torch.randn(2, 10, 512, requires_grad=True)
    y = activation_quant(x)

    # Create scalar loss and trigger backprop through the graph
    loss = y.sum()
    loss.backward()

    # Verify gradients reached x via the computational graph
    assert x.grad is not None, "No gradients received"
    assert not torch.isnan(x.grad).any(), "NaN in gradients"
    assert not torch.isinf(x.grad).any(), "Inf in gradients"
    assert x.grad.abs().max() < 1000, "Gradient explosion detected"


def test_activation_quant_edge_cases():
    """
    Test quantization handles zeros, small values, and large values.
    """

    # All zeros - should produce zeros
    x_zeros = torch.zeros(2, 5, 64)
    y_zeros = activation_quant(x_zeros)
    assert (y_zeros == 0).all()

    # Very small values (should be preserved approximately)
    x_small = torch.randn(2, 5, 64) * 1e-5
    y_small = activation_quant(x_small)
    assert not torch.isnan(y_small).any()
    assert not torch.isinf(y_small).any()

    # Very large values (should be clamped to int8 range)
    x_large = torch.randn(2, 5, 64) * 1e5
    y_large = activation_quant(x_large)
    assert not torch.isnan(y_large).any()
    assert not torch.isinf(y_large).any()

def test_weight_quant_exact_ternary():
    """Test that output has exactly 3 unique values: {-α, 0, +α}."""
    
    # Craft weights that will definitely produce all 3 ternary values
    # Large positive → +1, large negative → -1, small → 0
    w = torch.tensor([
        [2.0, -2.0, 0.1],   # Will round to: +1, -1, 0
        [1.5, -1.5, 0.05],  # Will round to: +1, -1, 0
        [3.0, -0.5, 0.2],   # Will round to: +1, -1, 0
    ])
    
    w_quant = weight_quant(w)
    unique_vals = torch.unique(w_quant)
    
    # Should have EXACTLY 3 unique values
    assert len(unique_vals) == 3
    
    # One should be zero
    assert torch.any(torch.abs(unique_vals) < 1e-6), \
        "Expected one value to be zero"
    
    # Non-zero values should be symmetric: {-α, 0, +α}
    non_zero = unique_vals[unique_vals.abs() > 1e-6]
    assert len(non_zero) == 2
    assert torch.isclose(non_zero[0], -non_zero[1], rtol=1e-5)
