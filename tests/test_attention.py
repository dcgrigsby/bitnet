import math

import pytest
import torch

from bitnet.attention import get_rotary_freqs, apply_rotary_emb


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


