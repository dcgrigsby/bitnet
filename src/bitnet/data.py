from typing import Any, Iterator, cast

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class WikiTextDataLoader:
    """DataLoader for WikiText-2 from Hugging Face.

    Downloads and tokenizes WikiText-2, yielding batches of token sequences.

    Args:
        tokenizer: Tokenizer instance
        batch_size: Batch size
        seq_len: Sequence length
        num_steps: Number of batches to generate (if None, iterate through full dataset)
        split: Which split to use ('train', 'validation', 'test')
    """

    tokenizer: PreTrainedTokenizer
    batch_size: int
    seq_len: int
    num_steps: int | None
    split: str
    dataset: Any

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        seq_len: int = 32,
        num_steps: int | None = None,
        split: str = "train",
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_steps = num_steps
        self.split = split

        # Load dataset
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield batches of tokenized sequences."""

        step_count = 0
        buffer: list[int] = []

        for sample in self.dataset:
            text: str = sample["text"].strip()
            if not text:
                continue

            # Tokenize
            tokens: list[int] = self.tokenizer.encode(text)
            buffer.extend(tokens)

            # Emit batches when buffer is large enough
            while len(buffer) >= self.batch_size * self.seq_len:
                batch_tokens: list[int] = buffer[: self.batch_size * self.seq_len]
                buffer = buffer[self.batch_size * self.seq_len :]

                # Reshape into batch
                batch = torch.tensor(batch_tokens).reshape(self.batch_size, self.seq_len)
                yield batch

                step_count += 1
                if self.num_steps is not None and step_count >= self.num_steps:
                    return

        # Emit final partial batch if we have steps remaining
        if self.num_steps is None or step_count < self.num_steps:
            if len(buffer) >= self.seq_len:
                remaining_tokens: list[int] = buffer[: self.batch_size * self.seq_len]
                if len(remaining_tokens) > 0:
                    # Pad if needed
                    if len(remaining_tokens) < self.batch_size * self.seq_len:
                        pad_len = self.batch_size * self.seq_len - len(remaining_tokens)
                        pad_token = cast(int, self.tokenizer.pad_token_id)
                        remaining_tokens.extend([pad_token] * pad_len)

                    batch = torch.tensor(remaining_tokens).reshape(self.batch_size, self.seq_len)
                    yield batch
