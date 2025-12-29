# Tic-Tac-Toe Experiment Design

**Date:** 2025-12-28
**Goal:** Train BitNet model to play tic-tac-toe by learning from synthetic game data

## Overview

Following the successful arithmetic experiment, we'll train a 12M parameter BitNet model to play tic-tac-toe. The model will learn game rules, valid moves, and winning conditions implicitly from training data - no hard-coded game logic during training.

## Data Representation

### Game Format

Sequential notation showing board state and move sequence:

```
_ _ _ _ _ _ _ _ _ | X:4 | O:0 | X:8 | O:2 | X:6 | Result: X wins
```

**Components:**
- Initial board: `_ _ _ _ _ _ _ _ _` (9 positions, 0-8 from top-left to bottom-right)
- Move notation: `| X:4` (player X plays position 4)
- Game result: `| Result: X wins` or `| Result: draw`

**Board Position Mapping:**
```
0 | 1 | 2
---+---+---
3 | 4 | 5
---+---+---
6 | 7 | 8
```

### Tokenizer Design

Character-level tokenizer (32 vocab size, matching arithmetic):

- **Digits:** `0-9` (tokens 0-9)
- **Players:** `X`, `O` (tokens 10-11)
- **Empty cell:** `_` (token 12)
- **Separators:** `|`, `:`, ` ` (tokens 13-15)
- **Result text:** `R`, `e`, `s`, `u`, `l`, `t`, `w`, `i`, `n`, `d`, `r`, `a` (tokens 16-27)
- **Special:** `<pad>`, `<s>`, `</s>`, `<unk>` (tokens 28-31)

## Training Data Generation

### Random Play Generator

Generate games where both players make random valid moves:

```python
class TicTacToeGame:
    def __init__(self):
        self.board = ['_'] * 9
        self.current_player = 'X'
        self.moves = []

    def get_valid_moves(self) -> list[int]:
        return [i for i in range(9) if self.board[i] == '_']

    def make_move(self, position: int) -> None:
        self.board[position] = self.current_player
        self.moves.append((self.current_player, position))
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self) -> str | None:
        # Check 3 rows, 3 columns, 2 diagonals
        # Return 'X', 'O', 'draw', or None

    def to_string(self) -> str:
        # Format as: "_ _ _ ... | X:4 | ... | Result: X wins"
```

**Game Statistics:**
- Average game length: 6-7 moves
- ~50 tokens per game (including board, moves, result)
- Win distribution: ~58% X wins, ~29% O wins, ~13% draws (random play)

### DataLoader

Similar to `ArithmeticDataLoader`:
- Generate games on-the-fly
- Tokenize and pack multiple games per sequence
- Yield batches of shape `(batch_size, seq_len)`
- Sequence length: 128 tokens (fits 2-3 games)

## Model Architecture

### Configuration

**BitNet 12M parameters** (scaled from 5M arithmetic):

```python
BitNetConfigTicTacToe(
    vocab_size=32,
    hidden_size=384,      # vs 256 for arithmetic
    num_layers=6,         # vs 4 for arithmetic
    num_heads=6,
    num_kv_heads=3,
    ffn_hidden_size=1152, # vs 768 for arithmetic
    max_seq_length=128,   # vs 64 for arithmetic
)
```

**Rationale for 12M:**
- Tic-tac-toe requires board state tracking and spatial reasoning
- More complex than arithmetic pattern matching
- Less complex than natural language (TinyStories)
- 12M sits between arithmetic (5M) and tinystories (12M) complexity

### Training Configuration

- **Steps:** 15,000 (vs 10,000 for arithmetic)
- **Batch size:** 32
- **Sequence length:** 128
- **Learning rate:** Two-stage schedule (same as arithmetic)
- **Loss:** CrossEntropyLoss for next-token prediction
- **Expected runtime:** ~60 minutes on RTX 3060

## Training & Evaluation

### Training Loop

Standard next-token prediction:
1. Model sees game sequence
2. Predicts next token at each position
3. CrossEntropyLoss on predictions vs ground truth
4. Model learns patterns: valid positions, alternating players, win conditions

### Evaluation During Training

Every 500 steps, test model by generating games:

```python
def test_gameplay(model, tokenizer, device):
    # Start new game: "_ _ _ _ _ _ _ _ _"
    # Generate moves using model (with masking)
    # Track: valid moves, game completion, results
```

**Metrics:**
- Valid move rate (should be 100% with masking)
- Game completion rate (reaches "Result:")
- Move distribution (does it favor center/corners?)

### Expected Results

- **Loss:** Drop from ~3.5 to ~0.5-1.0
- **Valid moves:** 100% (enforced by masking during generation)
- **Rule learning:** Model learns alternating players, stopping at win/draw
- **Strategy:** May learn basic heuristics (center, corners) from win patterns

## Interactive Play Interface

### Play Script

`play_tictactoe.py` - Interactive game against trained model:

```python
def display_board(board):
    # Pretty print 3x3 grid with position numbers

def get_human_move(board):
    # Prompt for move, validate it's legal

def get_model_move(model, tokenizer, game_sequence, board):
    # Encode current game sequence
    # Generate next move tokens: "| X:" or "| O:"
    # Then generate digit with logit masking for taken positions
    # Return predicted position
```

**Logit Masking:**
During generation, mask invalid moves:
```python
# Get logits for next position digit
logits = model(encoded_sequence)[:, -1, :]

# Mask taken positions
for pos in taken_positions:
    logits[0, pos] = float('-inf')

# Sample from valid positions only
next_pos = logits.argmax()
```

### Game Loop

1. Human chooses X or O
2. Loop until game ends:
   - Display board
   - If human turn: get input and validate
   - If model turn: generate with masking
   - Update board and check winner
3. Display final board and result

## Implementation Plan

### Files to Create

```
experiments/tictactoe/
├── train_bitnet_tictactoe.py          # Training script
├── start_tictactoe_experiment.sh      # Launcher
├── play_tictactoe.py                  # Interactive play
└── TICTACTOE_CONFIG.md                # Experiment documentation

src/bitnet/
├── data_tictactoe.py                  # Tokenizer, game generator, dataloader
└── config_tictactoe.py                # 12M model config
```

### Implementation Order

1. **data_tictactoe.py** - Tokenizer, game logic, dataloader
2. **config_tictactoe.py** - 12M parameter configuration
3. **train_bitnet_tictactoe.py** - Training script with game testing
4. **start_tictactoe_experiment.sh** - Launcher script
5. **play_tictactoe.py** - Interactive play interface
6. **justfile** - Add tasks: `exp-tictactoe`, `play-tictactoe`

### Testing Strategy

1. Verify tokenizer encode/decode with sample games
2. Generate 10 random games, inspect format
3. Train for 1000 steps, verify loss decreases
4. Test model generates valid moves (with masking)
5. Full training run (15k steps)
6. Play test games against model

## Success Criteria

**Minimum (Infrastructure Validation):**
- Loss drops below 1.0
- Model generates valid moves 100% of time (with masking)
- Model completes games (reaches "Result:")

**Target (Learned Game Rules):**
- Model plays legal games without masking
- Model recognizes win conditions
- Model shows basic strategy (center/corner preference)

**Stretch (Strategic Play):**
- Model never loses against random play
- Model demonstrates blocking behavior
- Model attempts to create winning patterns

## Future Extensions

After successful training:
1. Fine-tune on optimal play data (minimax games)
2. Train larger model (25M-50M) for better strategy
3. Add commentary generation (explain moves)
4. Extend to Connect Four or other grid games
