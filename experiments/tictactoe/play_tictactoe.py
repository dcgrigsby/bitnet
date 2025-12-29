#!/usr/bin/env python3
"""
Interactive tic-tac-toe game against trained BitNet model.

Usage:
    python play_tictactoe.py <checkpoint_path>

Example:
    python play_tictactoe.py runs/bitnet_12M_tictactoe_1234567/checkpoints/step_015000/checkpoint.pt
"""

import argparse
import sys
from pathlib import Path

import torch

from bitnet.config_tictactoe import BitNetConfigTicTacToe
from bitnet.data_tictactoe import TicTacToeTokenizer
from bitnet.transformer import BitNetModel


def display_board(board: list[str]) -> None:
    """Pretty print 3x3 board with position numbers."""
    print("\nCurrent Board:")
    print(f" {board[0]} | {board[1]} | {board[2]}     Position Guide:")
    print("---+---+---     0 | 1 | 2")
    print(f" {board[3]} | {board[4]} | {board[5]}     ---+---+---")
    print("---+---+---     3 | 4 | 5")
    print(f" {board[6]} | {board[7]} | {board[8]}     ---+---+---")
    print("            6 | 7 | 8")


def check_winner(board: list[str]) -> str | None:
    """Check if there's a winner or draw.

    Returns:
        'X' if X wins, 'O' if O wins, 'draw' if board full, None if game continues
    """
    # Check rows
    for i in range(0, 9, 3):
        if board[i] == board[i + 1] == board[i + 2] != "_":
            return board[i]

    # Check columns
    for i in range(3):
        if board[i] == board[i + 3] == board[i + 6] != "_":
            return board[i]

    # Check diagonals
    if board[0] == board[4] == board[8] != "_":
        return board[0]
    if board[2] == board[4] == board[6] != "_":
        return board[2]

    # Check for draw (board full)
    if "_" not in board:
        return "draw"

    return None


def get_human_move(board: list[str]) -> int:
    """Prompt human for move, validate it's legal."""
    while True:
        try:
            move_str = input("\nYour move (0-8): ").strip()
            if not move_str:
                continue
            move = int(move_str)
            if 0 <= move <= 8 and board[move] == "_":
                return move
            print("Invalid move! Position taken or out of range.")
        except (ValueError, KeyboardInterrupt):
            print("\nPlease enter a number 0-8")
        except EOFError:
            print("\nExiting...")
            sys.exit(0)


def get_model_move(
    model: BitNetModel,
    tokenizer: TicTacToeTokenizer,
    game_sequence: str,
    board: list[str],
    current_player: str,
    device: torch.device,
) -> int | None:
    """Generate model's next move with masking for valid positions only.

    Args:
        model: BitNet model
        tokenizer: Tokenizer
        game_sequence: Current game sequence string
        board: Current board state
        current_player: 'X' or 'O'
        device: Torch device

    Returns:
        Position (0-8) or None if generation failed
    """
    model.eval()

    with torch.no_grad():
        # Encode current game sequence
        input_ids = torch.tensor([tokenizer.encode(game_sequence)]).to(device)

        # Generate move sequence: " | X:4"
        # We need to generate: space, pipe, space, player, colon, digit
        generated = input_ids.clone()

        # Generate tokens one by one
        expected_sequence = f" | {current_player}:"
        for expected_char in expected_sequence:
            logits = model(generated)
            next_token_logits = logits[:, -1, :].squeeze()

            # For structured parts, we can guide or just take argmax
            # For simplicity, just take argmax and hope model learned the format
            next_token = next_token_logits.argmax().unsqueeze(0).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)

        # Now generate the position digit with masking
        logits = model(generated)
        next_token_logits = logits[:, -1, :].squeeze()

        # Mask invalid positions (taken squares)
        taken_positions = [i for i in range(9) if board[i] != "_"]
        for pos in taken_positions:
            # Mask the token ID for this digit
            digit_token_id = pos  # tokens 0-8 are digits 0-8
            next_token_logits[digit_token_id] = float("-inf")

        # Also boost valid positions slightly (optional)
        valid_positions = [i for i in range(9) if board[i] == "_"]
        if not valid_positions:
            return None

        # Sample from valid positions
        # Convert logits to probabilities for valid positions
        valid_logits = torch.tensor([next_token_logits[i].item() for i in valid_positions])
        if valid_logits.max() == float("-inf"):
            # Fallback: random valid move
            return valid_positions[0] if valid_positions else None

        # Take highest probability valid position
        best_valid_idx = valid_logits.argmax().item()
        position = valid_positions[best_valid_idx]

        model.train()
        return position


def play_game(checkpoint_path: str) -> None:
    """Play tic-tac-toe game against trained model."""
    # Load model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = BitNetConfigTicTacToe()
    model = BitNetModel(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from step {checkpoint['step']}")
    print(f"Device: {device}")

    tokenizer = TicTacToeTokenizer()

    # Game setup
    print("\n" + "=" * 50)
    print("Tic-Tac-Toe: Human vs BitNet")
    print("=" * 50)

    while True:
        choice = input("\nDo you want to play as X (first) or O (second)? (X/O): ").strip().upper()
        if choice in ["X", "O"]:
            human_player = choice
            model_player = "O" if choice == "X" else "X"
            break
        print("Please enter X or O")

    print(f"\nYou are {human_player}, Model is {model_player}")
    print(f"{'You' if human_player == 'X' else 'Model'} will go first.")

    # Initialize game
    board = ["_"] * 9
    current_player = "X"
    game_sequence = " ".join(["_"] * 9)  # Initial board

    # Game loop
    while True:
        display_board(board)

        # Check for winner
        result = check_winner(board)
        if result:
            if result == "draw":
                print("\nGame Over: It's a draw!")
            else:
                winner_name = "You" if result == human_player else "Model"
                print(f"\nGame Over: {winner_name} ({result}) wins!")
            break

        # Get move
        if current_player == human_player:
            print(f"\n{human_player}'s turn (You)")
            position = get_human_move(board)
        else:
            print(f"\n{model_player}'s turn (Model)")
            print("Model is thinking...")
            position = get_model_move(model, tokenizer, game_sequence, board, current_player, device)

            if position is None:
                print("Model failed to generate valid move!")
                print("Possible issues: model not trained enough or generation error")
                break

            print(f"Model plays position {position}")

        # Make move
        board[position] = current_player
        game_sequence += f" | {current_player}:{position}"

        # Switch player
        current_player = "O" if current_player == "X" else "X"

    # Ask to play again
    print("\n" + "=" * 50)
    play_again = input("Play again? (y/n): ").strip().lower()
    if play_again == "y":
        play_game(checkpoint_path)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Play tic-tac-toe against trained BitNet model")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    try:
        play_game(str(checkpoint_path))
    except KeyboardInterrupt:
        print("\n\nThanks for playing!")
        sys.exit(0)


if __name__ == "__main__":
    main()
