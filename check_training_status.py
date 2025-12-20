#!/usr/bin/env python3
"""
Check training status for a BitNet run.

Usage:
    python check_training_status.py runs/bitnet_95M_fineweb_T1.97B_1703097600

Reads metrics and samples from the run directory to show current progress.
Non-intrusive: does not interfere with running training.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def format_number(n: int | float) -> str:
    """Format large numbers with commas."""
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def read_latest_metrics(run_dir: Path) -> dict[str, Any] | None:
    """Read the latest metrics from scalars.jsonl."""
    scalars_file = run_dir / "metrics" / "scalars.jsonl"

    if not scalars_file.exists():
        return None

    # Read last line
    with open(scalars_file, "r") as f:
        lines = f.readlines()
        if not lines:
            return None
        return json.loads(lines[-1])


def read_latest_eval(run_dir: Path) -> dict[str, Any] | None:
    """Read the latest evaluation results."""
    eval_file = run_dir / "eval" / "eval_results.jsonl"

    if not eval_file.exists():
        return None

    # Read last line
    with open(eval_file, "r") as f:
        lines = f.readlines()
        if not lines:
            return None
        return json.loads(lines[-1])


def read_latest_samples(run_dir: Path, n: int = 3) -> list[dict[str, Any]]:
    """Read the latest N samples."""
    samples_file = run_dir / "samples" / "samples.jsonl"

    if not samples_file.exists():
        return []

    # Read last N lines
    with open(samples_file, "r") as f:
        lines = f.readlines()
        if not lines:
            return []

        samples = []
        for line in lines[-n:]:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return samples


def read_config(run_dir: Path) -> dict[str, Any] | None:
    """Read training configuration."""
    config_file = run_dir / "meta" / "config.json"

    if not config_file.exists():
        return None

    with open(config_file, "r") as f:
        return json.load(f)


def main() -> None:
    """Main status checking function."""
    parser = argparse.ArgumentParser(description="Check BitNet training status")
    parser.add_argument("run_dir", type=str, help="Path to run directory")
    parser.add_argument(
        "--watch",
        "-w",
        action="store_true",
        help="Watch mode: refresh every 10 seconds",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return

    # Read configuration
    config = read_config(run_dir)
    if not config:
        print(f"Error: Could not read config from {run_dir}")
        return

    total_steps = config["training"]["total_steps"]
    total_tokens = config["training"]["total_tokens"]
    batch_size = config["training"]["batch_size"]
    seq_len = config["training"]["seq_len"]

    # Read latest metrics
    metrics = read_latest_metrics(run_dir)
    if not metrics:
        print(f"No metrics found yet in {run_dir}")
        return

    # Extract values
    current_step = metrics["step"]
    tokens_seen = metrics["tokens_seen"]
    loss = metrics["loss"]
    lr = metrics["lr"]
    wd = metrics["wd"]
    grad_norm = metrics["grad_norm"]
    step_time_ms = metrics["step_time_ms"]
    tokens_per_sec = metrics["tokens_per_sec"]

    # Calculate progress
    progress_pct = (current_step / total_steps) * 100
    tokens_progress_pct = (tokens_seen / total_tokens) * 100

    # Estimate ETA
    steps_remaining = total_steps - current_step
    eta_seconds = (steps_remaining * step_time_ms) / 1000

    # Determine stage
    if current_step < total_steps // 2:
        stage = "Stage 1 (Regularization, WD=0.1)"
    else:
        stage = "Stage 2 (Convergence, WD=0.0)"

    # Print status
    print()
    print("=" * 80)
    print(f"Training Status: {run_dir.name}")
    print("=" * 80)
    print()

    print(f"Progress:  {current_step:,} / {total_steps:,} steps ({progress_pct:.1f}%)")
    print(
        f"Tokens:    {format_number(tokens_seen)} / {format_number(total_tokens)} ({tokens_progress_pct:.1f}%)"
    )
    print()

    print(f"Current Metrics (Step {current_step:,}):")
    print(f"  Loss:        {loss:.4f}")
    print(f"  LR:          {lr:.6f}")
    print(f"  WD:          {wd:.2f}")
    print(f"  Grad Norm:   {grad_norm:.4f}")
    print(f"  Throughput:  {format_number(tokens_per_sec)} tokens/sec")
    print(f"  Step Time:   {step_time_ms:.1f}ms")
    print()

    if "gpu_mem_allocated_gb" in metrics:
        print(f"GPU Memory:")
        print(f"  Allocated:   {metrics['gpu_mem_allocated_gb']:.2f} GB")
        print(f"  Reserved:    {metrics['gpu_mem_reserved_gb']:.2f} GB")
        print()

    print(f"Stage: {stage}")
    print(f"ETA:   {format_time(eta_seconds)}")
    print()

    # Print evaluation results if available
    eval_results = read_latest_eval(run_dir)
    if eval_results:
        print(f"Last Eval (Step {eval_results['step']:,}):")
        print(f"  Eval Loss:       {eval_results['eval_loss']:.4f}")
        print(f"  Train Loss:      {eval_results['train_loss']:.4f}")
        print(f"  Train-Eval Gap:  {eval_results['train_eval_gap']:.4f}")
        print()

    # Print recent samples
    samples = read_latest_samples(run_dir, n=3)
    if samples:
        print(f"Recent Samples (Step {samples[0]['step']:,}):")
        print()

        # Group by prompt
        by_prompt: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            prompt = sample["prompt"]
            if prompt not in by_prompt:
                by_prompt[prompt] = []
            by_prompt[prompt].append(sample)

        # Show first prompt only to keep output concise
        for prompt, prompt_samples in list(by_prompt.items())[:1]:
            print(f"  Prompt: \"{prompt}\"")
            for sample in prompt_samples:
                mode = sample["mode"]
                text = sample["generated_text"]
                entropy = sample["mean_entropy"]
                max_rep = sample["max_repetition"]

                print(f"    [{mode}] {text[:80]}...")
                print(
                    f"            (entropy={entropy:.2f}, max_rep={max_rep}, unique={sample['unique_tokens']})"
                )
            print()

    print("=" * 80)
    print()

    # Check for anomalies
    anomalies_dir = run_dir / "anomalies"
    if anomalies_dir.exists():
        anomaly_dirs = list(anomalies_dir.iterdir())
        if anomaly_dirs:
            print(f"⚠️  {len(anomaly_dirs)} anomalies detected:")
            for anom_dir in anomaly_dirs[-3:]:  # Show last 3
                print(f"    - {anom_dir.name}")
            print()


if __name__ == "__main__":
    main()
