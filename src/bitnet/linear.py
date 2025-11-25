import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from bitnet.quant import activation_quant, weight_quant


class BitLinear(nn.Linear):
    """BitLinear layer for BitNet b1.58.

    Replaces standard nn.Linear with ternary weight quantization
    and 8-bit activation quantization during training.
    """

    def __init__(self, in_features: int, out_features: int, norm_eps: float = 1e-5):
        super().__init__(in_features, out_features, bias=False)
        
        self.norm_eps = norm_eps

        # Pre-linear normalization
        self.norm = RMSNorm(in_features, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization.

        Args:
            x: Input tensor of shape [batch_size, seq_len, in_features]

        Returns:
            Output tensor of shape [batch_size, seq_len, out_features]
        """

        # RMSNorm (preserves variance for stable quantization)
        x_norm = self.norm(x)

        # Quantize weight with STE
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()

        # Backward: gradients bypass quantization (via detach)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

        # Standard linear operation
        y = F.linear(x_quant, w_quant, self.bias)

        return y
