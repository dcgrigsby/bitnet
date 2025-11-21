default:
    @just --list

test *args:
    uv run pytest {{ args }}
