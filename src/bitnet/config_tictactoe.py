"""Tic-tac-toe model configuration."""

from dataclasses import dataclass


@dataclass
class BitNetConfigTicTacToe:
    """12M parameter BitNet for tic-tac-toe game learning.

    Medium-sized model with minimal vocabulary (32 tokens) to learn
    game rules, valid moves, and strategy from synthetic game data.

    Architecture scaled from TinyStories (12M) but with smaller vocab
    and moderate sequence length for game sequences.
    """

    vocab_size: int = 32  # Digits, players, separators, result text, special tokens
    hidden_size: int = 384  # 12M params
    num_layers: int = 6
    num_heads: int = 6
    num_kv_heads: int = 3  # GQA 2:1 ratio
    ffn_hidden_size: int = 1152  # 3× expansion
    max_seq_length: int = 128  # Fits 2-3 complete games
    norm_eps: float = 1e-5

    # Training hyperparameters
    learning_rate: float = 0.0015  # BitNet standard
    weight_decay: float = 0.1  # Stage 1
    warmup_steps: int = 200
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    @property
    def head_dim(self) -> int:
        """Head dimension."""
        return self.hidden_size // self.num_heads
