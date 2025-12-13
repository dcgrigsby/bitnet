default:
    @just --list

test *args:
    uv run pytest {{ args }}

train-simple:
    uv run python examples/train_simple.py
