import math

import torch
import torch.nn as nn
from torch.nn import RMSNorm

from bitnet.linear import BitLinear


def get_rotary_freqs(
    head_dim: int, seq_len: int, theta: float = 10000.0, device: torch.device = None
) -> torch.tensor:
    """Compute rotary embedding frequencies.

    Args:
        head_dim: Head dimension
        seq_len: Sequence length
        theta: Frequency base
        device: Device to place tensor on

    Returns:
        Frequency tensor of shape [seq_len, head_dim//2]
    """

    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # Compute positional frequencies
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    return freqs


def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings.

    Args:
        x: Tensor of shape [batch_size, num_heads, seq_len, head_dim]
        freqs: Frequency tensor of shape [seq_len, head_dim//2]

    Returns:
        Rotated tensor of same shape
    """

    # Split into real and imaginary parts
    x_real = x[..., : x.shape[-1] // 2]
    x_imag = x[..., x.shape[-1] // 2 :]

    # Apply rotation: (a + bi) * (cost(theta) + i*sin(theta))
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]

    x_rot_real = x_real * cos - x_imag * sin
    x_rot_imag = x_real * sin + x_imag * cos

    return torch.cat([x_rot_real, x_rot_imag], dim=-1)
