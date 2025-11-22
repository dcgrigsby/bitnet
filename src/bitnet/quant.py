"""Quantization functions for BitNet b1.58.

Provides per-token 8-bit activation quantization and per-tensor ternary weight quantization.
"""
import torch


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """
    Per-token quantization to 8 bits.

    Quantizes activations per token (per row) to int8, then dequantizes.
    This is used during training.

    Args:
        x: Activation tensor of shape [batch_size, seq_len, hidden_size] or 
           [batch_size, hidden_size]

    Returns:
        Dequantized tensor of same shape as input
    """

    # Find max absolute value per token (last dimension is kept)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)

    # Quantize to int8 (still float dtype for autograd)
    x_int8 = (x * scale).round().clamp(-128, 127)

    # Dequantize
    x_dequant = x_int8 / scale

    return x_dequant

def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """
    Per-tensor quantization to 1.58 bits.

    Quantizes weights per tensor to ternary {-1, 0, 1}.
    Uses absmean quantization as per the paper.

    Args:
        w: Weight tensor of shape [out_features, in_features]

    Returns:
        Ternary tensor of shape [out_features, in_features]
    """

    # Compute scale as mean absolute value
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    
    # Scale, round, and clip to {-1, 0, 1}
    w_ternary = (w * scale).round().clamp(-1, 1)
    
    # Dequantize
    w_dequant = w_ternary / scale
    
    return w_dequant
