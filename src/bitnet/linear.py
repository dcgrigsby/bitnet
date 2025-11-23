import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from bitnet.quant import activation_quant, weight_quant


class BitLinear(nn.Linear):
    """BitLinear layer for BitNet b1.58

    Replaces standard nn.Linear with ternary weight quantization
    and 8-bit activation quantization during training.
    """

    def __init__(self, in_features: int, out_features: int):
        # Bitnet does not use bias
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization.

        Args:
            x: Input tensor of shape [batch_size, seq_len, in_features]

        Returns:
            Output tensor of shape [batch_size, seq_len, out_features]
        """

        # Quantize activation
        x_quant = x + (activation_quant(x) - x).detach()

        # Quantize weight
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()

        # Linear operation
        return F.linear(x_quant, w_quant, None)
