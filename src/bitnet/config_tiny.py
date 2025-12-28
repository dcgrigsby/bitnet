"""Tiny model configurations for validation experiments."""

from dataclasses import dataclass


@dataclass
class BitNetConfigTiny:
    """12M parameter BitNet for TinyStories validation.

    Small model with 2K vocabulary to test training infrastructure
    on simple natural language tasks.
    """

    vocab_size: int = 2048  # Custom TinyStories tokenizer
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 4  # GQA 2:1 ratio
    ffn_hidden_size: int = 1152  # 3× expansion
    max_seq_length: int = 256
    norm_eps: float = 1e-5

    # Training hyperparameters
    learning_rate: float = 0.0015  # BitNet standard
    weight_decay: float = 0.1  # Stage 1
    warmup_steps: int = 250
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    @property
    def head_dim(self) -> int:
        """Head dimension."""
        return self.hidden_size // self.num_heads


@dataclass
class BitNetConfigArithmetic:
    """5M parameter BitNet for arithmetic validation.

    Ultra-tiny model with minimal vocabulary to test that
    quantization and gradient flow work correctly.
    """

    vocab_size: int = 32  # Just digits + operators + special tokens
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2  # GQA 2:1 ratio
    ffn_hidden_size: int = 768  # 3× expansion
    max_seq_length: int = 64  # Short arithmetic expressions
    norm_eps: float = 1e-5

    # Training hyperparameters
    learning_rate: float = 0.0015
    weight_decay: float = 0.1
    warmup_steps: int = 100
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    @property
    def head_dim(self) -> int:
        """Head dimension."""
        return self.hidden_size // self.num_heads
