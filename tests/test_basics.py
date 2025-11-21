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

            if original_max >= 1e-4: # Skip near zero tokens
                # After quantization, max should be preserved (withing 2% with rounding)
                assert torch.isclose(quantized_max, original_max, rtol=0.02)

