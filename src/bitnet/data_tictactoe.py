"""DataLoader and generator for synthetic tic-tac-toe dataset."""

import random
from typing import Iterator

import torch


class TicTacToeTokenizer:
    """Character-level tokenizer for tic-tac-toe games.

    Vocabulary:
        - Digits: 0-9 (tokens 0-9)
        - Players: X O (tokens 10-11)
        - Empty: _ (token 12)
        - Separators: | : space (tokens 13-15)
        - Result text: R e s u l t w i n d r a (tokens 16-27)
        - Special tokens: <pad> <s> </s> <unk> (tokens 28-31)
    """

    def __init__(self) -> None:
        # Build vocabulary
        self.char_to_id = {str(i): i for i in range(10)}  # 0-9
        self.char_to_id.update({"X": 10, "O": 11})
        self.char_to_id["_"] = 12
        self.char_to_id["|"] = 13
        self.char_to_id[":"] = 14
        self.char_to_id[" "] = 15

        # Result text characters
        result_chars = "Resultwindr"  # Unique chars in "Result: X wins/draw"
        for i, char in enumerate(result_chars):
            if char not in self.char_to_id:
                self.char_to_id[char] = 16 + i

        # Special tokens
        self.char_to_id["<pad>"] = 28
        self.char_to_id["<s>"] = 29
        self.char_to_id["</s>"] = 30
        self.char_to_id["<unk>"] = 31

        # Reverse mapping
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

        self.vocab_size = 32
        self.pad_token_id = 28
        self.eos_token_id = 30

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
        special_tokens = {28, 29, 30, 31}  # <pad>, <s>, </s>, <unk>

        for tid in token_ids:
            if skip_special_tokens and tid in special_tokens:
                continue
            chars.append(self.id_to_char.get(tid, "<unk>"))

        return "".join(chars)


class TicTacToeGame:
    """Tic-tac-toe game with random play generation."""

    def __init__(self) -> None:
        self.board = ["_"] * 9  # positions 0-8
        self.current_player = "X"
        self.moves: list[tuple[str, int]] = []
        self.winner: str | None = None

    def get_valid_moves(self) -> list[int]:
        """Get list of available positions."""
        return [i for i in range(9) if self.board[i] == "_"]

    def make_move(self, position: int) -> None:
        """Make a move at the given position."""
        if self.board[position] != "_":
            raise ValueError(f"Position {position} already taken")

        self.board[position] = self.current_player
        self.moves.append((self.current_player, position))
        self.current_player = "O" if self.current_player == "X" else "X"

    def check_winner(self) -> str | None:
        """Check if there's a winner or draw.

        Returns:
            'X' if X wins, 'O' if O wins, 'draw' if board full, None if game continues
        """
        # Check rows
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i + 1] == self.board[i + 2] != "_":
                return self.board[i]

        # Check columns
        for i in range(3):
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != "_":
                return self.board[i]

        # Check diagonals
        if self.board[0] == self.board[4] == self.board[8] != "_":
            return self.board[0]
        if self.board[2] == self.board[4] == self.board[6] != "_":
            return self.board[2]

        # Check for draw (board full)
        if "_" not in self.board:
            return "draw"

        return None

    def to_string(self) -> str:
        """Convert game to string format.

        Format: _ _ _ _ _ _ _ _ _ | X:4 | O:0 | X:8 | Result: X wins
        """
        # Initial board state
        parts = [" ".join(["_"] * 9)]

        # Add moves
        for player, pos in self.moves:
            parts.append(f" | {player}:{pos}")

        # Add result
        if self.winner == "draw":
            parts.append(" | Result: draw")
        elif self.winner:
            parts.append(f" | Result: {self.winner} wins")

        return "".join(parts)


def generate_random_game() -> str:
    """Generate a random tic-tac-toe game.

    Returns:
        Game string in format: "_ _ _ ... | X:4 | ... | Result: X wins"
    """
    game = TicTacToeGame()

    while True:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            game.winner = "draw"
            break

        # Make random move
        move = random.choice(valid_moves)
        game.make_move(move)

        # Check for winner
        winner = game.check_winner()
        if winner:
            game.winner = winner
            break

    return game.to_string()


class TicTacToeDataLoader:
    """DataLoader for synthetic tic-tac-toe games.

    Generates games on-the-fly during training with random play.

    Args:
        batch_size: Batch size
        seq_len: Sequence length (packs multiple games per sequence)
        num_steps: Number of batches to generate
    """

    def __init__(
        self,
        batch_size: int = 32,
        seq_len: int = 128,
        num_steps: int = 15000,
    ) -> None:
        self.tokenizer = TicTacToeTokenizer()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_steps = num_steps

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield batches of tokenized tic-tac-toe games."""
        step_count = 0
        buffer: list[int] = []

        # Generate games continuously
        while step_count < self.num_steps:
            # Generate random game
            game_str = generate_random_game()

            # Tokenize
            tokens = self.tokenizer.encode(game_str, add_special_tokens=False)
            buffer.extend(tokens)

            # Add separator (space) between games
            buffer.append(self.tokenizer.char_to_id[" "])

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
