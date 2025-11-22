import pytest
import torch

from bitnet.quant import activation_quant, weight_quant


def test_activation_quant_per_token_scaling():
    """Test that each token is independently scaled to int8 range.

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
    """Test that quantization produces discrete, not continous, vales.

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
    """Test that gradients flow through quantization via STE.

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
    """Test quantization handles zeros, small values, and large values."""

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
    w = torch.tensor(
        [
            [2.0, -2.0, 0.1],  # Will round to: +1, -1, 0
            [1.5, -1.5, 0.05],  # Will round to: +1, -1, 0
            [3.0, -0.5, 0.2],  # Will round to: +1, -1, 0
        ]
    )

    w_quant = weight_quant(w)
    unique_vals = torch.unique(w_quant)

    # Should have EXACTLY 3 unique values
    assert len(unique_vals) == 3

    # One should be zero
    assert torch.any(torch.abs(unique_vals) < 1e-6), "Expected one value to be zero"

    # Non-zero values should be symmetric: {-α, 0, +α}
    non_zero = unique_vals[unique_vals.abs() > 1e-6]
    assert len(non_zero) == 2
    assert torch.isclose(non_zero[0], -non_zero[1], rtol=1e-5)


def test_weight_quant_sign_preservation():
    """Test ternary quantization preserves sign for large weights, zeros small weights."""

    torch.manual_seed(42)  # Reproducibility

    # Create controlled distribution
    w = torch.cat(
        [
            torch.full((500,), 3.0),  # Large positive
            torch.full((500,), -3.0),  # Large negative
            torch.randn(500) * 0.1,  # Small random (should mostly zero)
        ]
    )

    w_quant = weight_quant(w)

    # Test 1: Large weights preserve sign exactly
    large_pos = w_quant[w == 3.0]
    assert (large_pos > 0).all()

    large_neg = w_quant[w == -3.0]
    assert (large_neg < 0).all()

    # Test 2: Small weights have high zero rate
    small_mask = w.abs() < 0.5
    small_quant = w_quant[small_mask]
    zero_rate = (small_quant.abs() < 1e-6).float().mean()
    assert zero_rate > 0.7


def test_weight_quant_scale_factor():
    """Test that scale factor is computed correctly from absmean.

    BitNet uses absmean quantization: scale = 1 / mean(|w|)
    After quantization to {-1, 0, 1} and dequantization,
    the maximum absolute value should equal 1/scale = mean(|w|)
"""

    # Create weights with known, controlled mean absolute value
    # mean(|w|) = (4*2.0 + 4*1.0 + 2*0.0) / 10 = 12/10 = 1.2
    w = torch.tensor(
        [
            2.0,
            2.0,
            2.0,
            2.0,  # Large positive
            -1.0,
            -1.0,
            -1.0,
            -1.0,  # Medium negative
            0.1,
            -0.1,  # Small (will likely zero)
        ]
    )

    # Manually compute expected scale
    expected_absmean = w.abs().mean()  # Should be 1.2
    expected_scale = 1.0 / expected_absmean

    # Quantize
    w_quant = weight_quant(w)

    # After dequantization, max absolute value should equal 1/scale = absmean
    actual_max_abs = w_quant.abs().max()
    expected_max_abs = 1.0 / expected_scale  # = absmean = 1.2

    assert torch.isclose(actual_max_abs, expected_max_abs, rtol=0.01)

    # Verify the actual quantized levels are correct
    # Large weights (2.0) should map to +1/scale = 1.2 (or very close)
    large_positive_quant = w_quant[w == 2.0]
    assert torch.allclose(
        large_positive_quant, torch.full_like(large_positive_quant, 1.2), atol=0.05
    )

    # Medium negative weights (-1.0) should map to -1/scale ≈ -1.2 or 0 depending on rounding
    # Since -1.0 * (1/1.2) = -0.833, rounds to -1, then / scale = -1.2
    medium_negative_quant = w_quant[w == -1.0]
    # Should be either -1.2 (if rounded to -1) or 0 (if rounded to 0)
    assert (
        (medium_negative_quant == 0)
        | torch.isclose(medium_negative_quant, torch.tensor(-1.2), atol=0.05)
    ).all()


def test_weight_quant_scale_factor_simple():
    """Simplified test: verify scale factor with perfectly controlled weights."""

    # Use weights where absmean = 1.0 for easy calculation
    w = torch.tensor([1.0, 1.0, -1.0, -1.0])  # absmean = 1.0

    w_quant = weight_quant(w)

    # With absmean = 1.0, scale = 1.0
    # Quantized values should be exactly {-1.0, 0.0, 1.0}
    expected_scale = 1.0
    expected_max = 1.0 / expected_scale  # = 1.0

    assert w_quant.abs().max() == expected_max

    # All values should be ±1.0 (since original is ±1.0 and scale is 1.0)
    # ±1.0 * 1.0 = ±1.0, round to ±1, then /1.0 = ±1.0
    assert torch.allclose(w_quant, w, atol=0.01)


def test_weight_quant_scale_factor_known_mean():
    """Test scale factor with various known mean absolute values."""

    test_cases = [
        # (weights, expected_absmean, expected_max_abs)
        (torch.tensor([2.0, 2.0, 2.0, 2.0]), 2.0, 2.0),
        (torch.tensor([0.5, 0.5, 0.5, 0.5]), 0.5, 0.5),
        (torch.tensor([3.0, -3.0, 1.0, -1.0]), 2.0, 2.0),
    ]

    for w, expected_absmean, expected_max_abs in test_cases:
        # Verify our test case setup
        actual_absmean = w.abs().mean()
        assert torch.isclose(actual_absmean, torch.tensor(expected_absmean), atol=1e-6)

        # Quantize
        w_quant = weight_quant(w)

        # Max absolute quantized value should equal absmean (= 1/scale)
        actual_max = w_quant.abs().max()
        assert torch.isclose(actual_max, torch.tensor(expected_max_abs), rtol=0.02)


def test_weight_quant_scale_computation_directly():
    """Test the scale computation formula directly."""

    # Create deterministic weights
    torch.manual_seed(42)
    w = torch.randn(100, 50)

    # Compute scale manually (matching implementation)
    expected_scale = 1.0 / w.abs().mean().clamp(min=1e-5)

    # Quantize
    w_quant = weight_quant(w)

    # The quantized values should only be at levels: {-1/scale, 0, +1/scale}
    unique_vals = torch.unique(w_quant)

    # Remove zero from unique values
    non_zero_unique = unique_vals[unique_vals.abs() > 1e-6]

    # Should have exactly 2 non-zero unique values (positive and negative)
    assert len(non_zero_unique) <= 2

    # These should be ±1/scale
    expected_magnitude = 1.0 / expected_scale
    for val in non_zero_unique:
        assert torch.isclose(val.abs(), expected_magnitude, rtol=0.01)
