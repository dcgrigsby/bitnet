"""DataLoader for FineWeb-Edu dataset with streaming support."""

import hashlib
from typing import Iterator

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class FineWebEduDataLoader:
    """Streaming DataLoader for FineWeb-Edu from Hugging Face.

    Downloads and tokenizes FineWeb-Edu in streaming mode, yielding batches
    of token sequences without requiring the full dataset download.

    Supports distributed training via rank/world_size sharding.

    Args:
        tokenizer: Tokenizer instance (e.g., LlamaTokenizer)
        batch_size: Batch size (per GPU when using DDP)
        seq_len: Sequence length
        num_steps: Number of batches to generate
        split: Which split to use (default: 'train')
        name: Dataset config name (default: 'sample-10BT')
        rank: Current rank for distributed training (default: 0)
        world_size: Total number of ranks (default: 1)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32,
        seq_len: int = 1024,
        num_steps: int = 60000,
        split: str = "train",
        name: str = "sample-10BT",  # Use 10BT sample (more stable than default)
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_steps = num_steps
        self.split = split
        self.name = name
        self.rank = rank
        self.world_size = world_size

        # Load dataset in streaming mode
        # Note: Some parquet files have different schemas (date column presence)
        # We handle this by only accessing the 'text' field
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=name,
            split=split,
            streaming=True,
            trust_remote_code=True,  # Allow handling schema variations
        )

        # For distributed training, use different shuffle seeds per rank
        # This ensures each rank processes different data
        if world_size > 1:
            # Each rank gets a different shuffle of the data
            # With large buffer_size + rank-specific seed, data overlap is minimal
            shuffle_seed = 42 + rank * 1000
            self.dataset = self.dataset.shuffle(seed=shuffle_seed, buffer_size=10000)

        # Cache first 1000 tokens for fingerprinting
        self._fingerprint_tokens: list[int] | None = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield batches of tokenized sequences."""

        step_count = 0
        buffer: list[int] = []

        # Store first 1000 tokens for fingerprinting
        fingerprint_collected = False
        fingerprint_buffer: list[int] = []

        for sample in self.dataset:
            text: str = sample["text"].strip()
            if not text:
                continue

            # Tokenize
            tokens: list[int] = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            # Collect fingerprint tokens
            if not fingerprint_collected:
                fingerprint_buffer.extend(tokens)
                if len(fingerprint_buffer) >= 1000:
                    self._fingerprint_tokens = fingerprint_buffer[:1000]
                    fingerprint_collected = True

            # Emit batches when buffer is large enough
            while len(buffer) >= self.batch_size * self.seq_len:
                batch_tokens: list[int] = buffer[: self.batch_size * self.seq_len]
                buffer = buffer[self.batch_size * self.seq_len :]

                # Reshape into batch
                batch = torch.tensor(batch_tokens, dtype=torch.long).reshape(
                    self.batch_size, self.seq_len
                )
                yield batch

                step_count += 1
                if step_count >= self.num_steps:
                    return

    def get_fingerprint(self) -> str:
        """Get SHA256 hash of first 1000 token IDs for reproducibility verification.

        Returns:
            Hex string of SHA256 hash

        Note:
            Must be called after at least one iteration to collect tokens.
        """
        if self._fingerprint_tokens is None:
            return "not_yet_collected"

        # Convert token IDs to bytes and hash
        token_bytes = b"".join(
            tok.to_bytes(4, byteorder="little") for tok in self._fingerprint_tokens
        )
        return hashlib.sha256(token_bytes).hexdigest()

    def get_fingerprint_tokens(self) -> list[int]:
        """Get the first 1000 token IDs for detailed fingerprinting.

        Returns:
            List of first 1000 token IDs, or empty list if not yet collected
        """
        return self._fingerprint_tokens or []
