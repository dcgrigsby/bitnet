"""DataLoader and generator for synthetic arithmetic dataset."""

import hashlib
import random
from typing import Iterator

import torch


class ArithmeticTokenizer:
    """Simple character-level tokenizer for arithmetic expressions.

    Vocabulary:
        - Digits: 0-9 (tokens 0-9)
        - Operators: + - * / (tokens 10-13)
        - Equals: = (token 14)
        - Space: (token 15)
        - Special tokens: <pad> <s> </s> <unk> (tokens 16-19)
    """

    def __init__(self) -> None:
        # Build vocabulary
        self.char_to_id = {str(i): i for i in range(10)}  # 0-9
        self.char_to_id.update({"+": 10, "-": 11, "*": 12, "/": 13})
        self.char_to_id["="] = 14
        self.char_to_id[" "] = 15
        self.char_to_id["<pad>"] = 16
        self.char_to_id["<s>"] = 17
        self.char_to_id["</s>"] = 18
        self.char_to_id["<unk>"] = 19

        # Reverse mapping
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

        self.vocab_size = 32  # Leave room for future expansion
        self.pad_token_id = 16
        self.eos_token_id = 18

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add <s> and </s>

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.char_to_id["<s>"])

        for char in text:
            tokens.append(self.char_to_id.get(char, self.char_to_id["<unk>"]))

        if add_special_tokens:
            tokens.append(self.char_to_id["</s>"])

        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        chars = []
        special_tokens = {16, 17, 18, 19}  # <pad>, <s>, </s>, <unk>

        for tid in token_ids:
            if skip_special_tokens and tid in special_tokens:
                continue
            chars.append(self.id_to_char.get(tid, "<unk>"))

        return "".join(chars)


def generate_arithmetic_sample(max_value: int = 100) -> str:
    """Generate a random arithmetic expression with answer.

    Args:
        max_value: Maximum value for operands

    Returns:
        Arithmetic expression string (e.g., "25 + 17 = 42")
    """
    a = random.randint(0, max_value)
    b = random.randint(1, max_value)  # Avoid division by zero
    op = random.choice(["+", "-", "*", "/"])

    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    else:  # division
        result = a // b  # Integer division

    return f"{a} {op} {b} = {result}"


class ArithmeticDataLoader:
    """DataLoader for synthetic arithmetic expressions.

    Generates simple arithmetic problems on-the-fly during training.

    Args:
        batch_size: Batch size
        seq_len: Sequence length (should be long enough for expressions)
        num_steps: Number of batches to generate
        max_value: Maximum value for operands in expressions
    """

    def __init__(
        self,
        batch_size: int = 32,
        seq_len: int = 64,
        num_steps: int = 10000,
        max_value: int = 100,
    ) -> None:
        self.tokenizer = ArithmeticTokenizer()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_steps = num_steps
        self.max_value = max_value

        # Cache first 1000 tokens for fingerprinting
        self._fingerprint_tokens: list[int] | None = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield batches of tokenized arithmetic expressions."""
        step_count = 0
        buffer: list[int] = []

        # Store first 1000 tokens for fingerprinting
        fingerprint_collected = False
        fingerprint_buffer: list[int] = []

        # Generate samples continuously
        while step_count < self.num_steps:
            # Generate arithmetic expression
            expression = generate_arithmetic_sample(self.max_value)

            # Tokenize
            tokens = self.tokenizer.encode(expression, add_special_tokens=False)
            buffer.extend(tokens)

            # Add separator (space) between expressions
            buffer.append(self.tokenizer.char_to_id[" "])

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
