import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from typing_extensions import override

from bitnet.quant import activation_quant, weight_quant


class BitLinear(nn.Linear):
    """BitLinear layer for BitNet b1.58.

    Replaces standard nn.Linear with ternary weight quantization
    and 8-bit activation quantization during training.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        norm_eps: RMSNorm epsilon
        disable_quant: If True, skip quantization and use standard Linear
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm_eps: float = 1e-5,
        disable_quant: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=False)

        self.norm_eps: float = norm_eps
        self.disable_quant: bool = disable_quant

        # Pre-linear normalization
        self.norm: RMSNorm = RMSNorm(in_features, eps=norm_eps)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional quantization.

        Args:
            input: Input tensor of shape [batch_size, seq_len, in_features]

        Returns:
            Output tensor of shape [batch_size, seq_len, out_features]
        """

        # RMSNorm (preserves variance for stable quantization)
        x_norm = self.norm(input)

        if self.disable_quant:
            # Standard linear without quantization
            y = F.linear(x_norm, self.weight, self.bias)
        else:
            # Quantize weight with STE
            w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()

            # Backward: gradients bypass quantization (via detach)
            x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

            # Standard linear operation
            y = F.linear(x_quant, w_quant, self.bias)

        return y
