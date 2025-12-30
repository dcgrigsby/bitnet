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

# Utilities
plot-loss run_id:
    uv run python scripts/plot_loss.py runs/{{run_id}}

check-status run_id *args:
    uv run python scripts/check_training_status.py runs/{{run_id}} {{args}}

chat model_path:
    uv run python scripts/chat_bitnet.py {{model_path}}
