default:
    @just --list

test *args:
    uv run pytest {{ args }}

train-simple steps='48':
    uv run python examples/train_simple.py {{steps}}
