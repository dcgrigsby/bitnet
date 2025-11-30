import math

import pytest
import torch

from bitnet.attention import Attention, apply_rotary_emb, get_rotary_freqs


def test_get_rotary_freqs_position_dependent():
    """Test rotary frequencies vary with position."""

    head_dim = 64
    seq_len = 20

    freqs = get_rotary_freqs(head_dim, seq_len)

    # Different positions should have different frequencies
    assert not torch.allclose(freqs[0], freqs[1])
    assert not torch.allclose(freqs[0], freqs[10])

    # Frequencies should be bounded (for numberical stability)
    assert not torch.isnan(freqs).any()
    assert not torch.isinf(freqs).any()


def test_apply_rotary_emb_actually_rotates():
    """Test rotary emeddings actually modify the input."""

    batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 64

    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    freqs = get_rotary_freqs(head_dim, seq_len)
    y = apply_rotary_emb(x, freqs)

    # Output should be different from input (rotation applied)
    assert not torch.allclose(x, y)

    # Magniture should be approximately preserved (rotation is norm preserving)
    x_norm = torch.norm(x, dim=-1)
    y_norm = torch.norm(y, dim=-1)

    assert torch.allclose(x_norm, y_norm, rtol=0.01)


def test_apply_rotary_emb_relative_position():
    """Test rotary embeddings encode relative position information."""

    batch_size, num_heads, seq_len, head_dim = 1, 1, 10, 64
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = q.clone()  # Same content

    freqs = get_rotary_freqs(head_dim, seq_len)
    q_rot = apply_rotary_emb(q, freqs)
    k_rot = apply_rotary_emb(k, freqs)

    # Compute attention scores (Q @ K^T)
    scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))

    # Diagonal (i==j) should have highest scores because:
    # - Content is identical (q=k) 
    # - Relative position is 0 (no rotation mismatch)
    diagonal_scores = torch.diagonal(scores[0, 0])
    off_diagonal_max = (
        scores[0, 0]
        .masked_fill(torch.eye(seq_len, dtype=torch.bool), float("-inf"))
        .max(dim=-1)
        .values
    )

    # Most positions should have higher score with themselves
    assert (diagonal_scores > off_diagonal_max).float().mean() > 0.6


def test_attention_forward_backward():
    """Test attention supports forward and backward."""
    hidden_size = 768
    num_heads = 12
    num_kv_heads = 12

    attn = Attention(hidden_size, num_heads, num_kv_heads)
    x = torch.randn(2, 10, hidden_size, requires_grad=True)
    y = attn(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert attn.qkv_proj.weight.grad is not None
    assert attn.out_proj.weight.grad is not None
    assert not torch.isnan(y).any()
