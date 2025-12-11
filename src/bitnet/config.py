from dataclasses import dataclass


@dataclass
class BitNetConfig:
    """Configuration for Bitnet b1.58 model."""

    vocab_size: int = 32000  # matches LLaMA tokenizer

    hidden_size: int = 768  # dimensions for each token

    num_layers: int = 12  # number of transformer layers

    num_heads: int = 12  # number of heads in each transformer layer

    num_kv_heads: int = 12  # number per heads per layer

    ffn_hidden_size: int = 3072  # inside each transformer

    max_seq_length: int = 2048

    norm_eps: float = 1e-5  # so you don't divide by zero

    learning_rate: float = 1.5e-3  # BitNet requires 5-7x higher LR (arXiv:2402.17764)

    weight_decay: float = 0.1  # first stage only

    warmup_steps: int = 375

    adam_beta1: float = 0.9

    adam_beta2: float = 0.95

    @property
    def head_dim(self) -> int:
        """Head dimension."""
        return self.hidden_size // self.num_heads

