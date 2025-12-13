from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn import RMSNorm
from typing_extensions import override

from bitnet.attention import Attention
from bitnet.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Single transformer block with attention + FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_hidden_size: int,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.attention: Attention = Attention(
            hidden_size,
            num_heads,
            num_kv_heads,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
        )

        self.feedforward: FeedForward = FeedForward(hidden_size, ffn_hidden_size, norm_eps=norm_eps)

    @override
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            attn_mask: Attention mask (optional)

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """

        # Attention with residual
        attn_output = self.attention(x, attn_mask)
        x = x + attn_output

        # FFN with residual
        ffn_output = self.feedforward(x)
        x = x + ffn_output

        return x


class BitNetModel(nn.Module):
    """BitNet b1.58 model"""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config: Any = config

        # Token embeddings
        self.token_embeddings: nn.Embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                TransformerBlock(
                    config.hidden_size,
                    config.num_heads,
                    config.num_kv_heads,
                    config.ffn_hidden_size,
                    norm_eps=config.norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Final normalization
        self.final_norm: RMSNorm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        # Output projection (language modeling head)
        self.lm_head: nn.Linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    @override
    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attn_mask: Attention mask (optional)

        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]]
        """

        # Embed tokens
        x = self.token_embeddings(input_ids)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits
