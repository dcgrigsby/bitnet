import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

from torch.nn import RMSNorm
from bitnet.linear import BitLinear


def get_rotary_freqs(
    head_dim: int, seq_len: int, theta: float = 10000.0, device: Optional[torch.device] = None
) -> torch.Tensor:
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


class Attention(nn.Module):
    """Multi-head self-attention with rotary embeddings."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        disable_quant: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads  # number of q (query) heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = hidden_size // num_heads
        self.rope_theta: float = rope_theta

        assert num_heads % num_kv_heads == 0
        self.heads_per_group: int = num_heads // num_kv_heads

        # Query, Key, Value projections (combined)
        self.qkv_proj: BitLinear = BitLinear(
            hidden_size, (num_heads + 2 * num_kv_heads) * self.head_dim, disable_quant=disable_quant
        )

        # Output projection
        self.out_proj: BitLinear = BitLinear(num_heads * self.head_dim, hidden_size, disable_quant=disable_quant)

        # Pre-attention normalization
        self.norm: RMSNorm = RMSNorm(hidden_size, eps=norm_eps)

    @override
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attention forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            attn_mask: Attention mask (optional)

        Returns:
            Output tensorof shape [batch_size, seq_len, hidden_size]
        """

        batch_size, seq_len, _ = x.shape

        # Apply normalization
        x_norm = self.norm(x)

        # Project to Q, K, V
        qkv = self.qkv_proj(x_norm)

        # Split into Q, K, V
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        q = qkv[..., :q_size]
        k = qkv[..., q_size : q_size + kv_size]
        v = qkv[..., q_size + kv_size :]

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        # Apply rotary embeddings
        freqs = get_rotary_freqs(self.head_dim, seq_len, self.rope_theta, x.device)
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)

        # Repeat K, V for grouped query attention
        if self.heads_per_group > 1:
            k = k.repeat_interleave(self.heads_per_group, dim=1)
            v = v.repeat_interleave(self.heads_per_group, dim=1)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask, or causal mask if none provided
        if attn_mask is not None:
            scores = scores + attn_mask
        else:
            # Default causal mask
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device),
                diagonal=1,
            )
            scores = scores + causal_mask

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, q_size)

        # Output projection
        output = self.out_proj(attn_output)

        return output
