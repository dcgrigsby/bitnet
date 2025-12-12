import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from typing_extensions import override

from bitnet.linear import BitLinear


class FeedForward(nn.Module):
    """Feed-forward layer with SwiGLU activation"""

    def __init__(self, hidden_size: int, ffn_hidden_size: int, norm_eps: float = 1e-5) -> None:
        super().__init__()

        self.hidden_size: int = hidden_size
        self.ffn_hidden_size: int = ffn_hidden_size

        # SwiGLU gate and up projection (combined)
        self.gate_up: BitLinear = BitLinear(hidden_size, 2 * ffn_hidden_size)

        # Down projection
        self.down: BitLinear = BitLinear(ffn_hidden_size, hidden_size)

        # Pre-FFN normalization
        self.norm: RMSNorm = RMSNorm(hidden_size, eps=norm_eps)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU.

        Args:
            x: Input tensor shape [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """

        # Apply normalization
        x_norm = self.norm(x)

        # Gate and up projections
        gate_up = self.gate_up(x_norm)
        gate, up = gate_up.chunk(2, dim=-1)

        # SwiGLU: use squared ReLu
        gate = F.relu(gate) ** 2

        # Combine gate and up
        hidden = gate * up

        # Down projection
        output = self.down(hidden)

        return output
