default:
    @just --list

test *args:
    uv run pytest {{ args }}

# Experiments - Run full experiment with default settings
exp-arithmetic:
    @bash experiments/arithmetic/start_arithmetic_experiment.sh

exp-tinystories:
    @bash experiments/tinystories/start_tinystories_experiment.sh

exp-baseline:
    @bash experiments/baseline-95m/start_training_400k.sh

# Direct training - Run training scripts with custom arguments
train-arithmetic *args:
    uv run python experiments/arithmetic/train_bitnet_arithmetic.py {{args}}

train-tinystories *args:
    uv run python experiments/tinystories/train_bitnet_tiny.py {{args}}

train-baseline *args:
    uv run python experiments/baseline-95m/train_bitnet_95m.py {{args}}

# Utilities
plot-loss run_id:
    uv run python scripts/plot_loss.py {{run_id}}

check-status run_id *args:
    uv run python scripts/check_training_status.py runs/{{run_id}} {{args}}

chat model_path:
    uv run python scripts/chat_bitnet.py {{model_path}}
