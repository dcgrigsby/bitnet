"""DataLoader for TinyStories dataset with custom tokenizer."""

import hashlib
from typing import Iterator

import torch
from datasets import load_dataset
from tokenizers import Tokenizer


class TinyStoriesDataLoader:
    """Streaming DataLoader for TinyStories from Hugging Face.

    Uses custom 2K vocabulary tokenizer trained on TinyStories.
    Downloads and tokenizes TinyStories in streaming mode.

    Args:
        tokenizer_path: Path to trained tokenizer JSON file
        batch_size: Batch size
        seq_len: Sequence length
        num_steps: Number of batches to generate
        split: Which split to use (default: 'train')
    """

    def __init__(
        self,
        tokenizer_path: str,
        batch_size: int = 32,
        seq_len: int = 256,
        num_steps: int = 30000,
        split: str = "train",
    ) -> None:
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_steps = num_steps
        self.split = split
        self.tokenizer_path = tokenizer_path

        # Load dataset in streaming mode
        self.dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )

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
            encoded = self.tokenizer.encode(text)
            tokens: list[int] = encoded.ids
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
        """Get SHA256 hash of first 1000 token IDs.

        Returns:
            Hex string of SHA256 hash
        """
        if self._fingerprint_tokens is None:
            return "not_yet_collected"

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
