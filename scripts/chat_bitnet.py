#!/usr/bin/env python3
"""
Interactive chat interface for BitNet 95M model.

Loads the trained model and provides a simple REPL for text generation
using typical sampling (temperature=0.9, top_p=0.95) matching Claude/ChatGPT style.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
from transformers import LlamaTokenizer, PreTrainedTokenizer

from src.bitnet.config import BitNetConfig
from src.bitnet.generation import generate_typical
from src.bitnet.transformer import BitNetModel


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Tuple[BitNetModel, BitNetConfig, int]:
    """Load model checkpoint and extract config.

    Args:
        checkpoint_path: Path to checkpoint.pt file
        device: Device to load model on

    Returns:
        Tuple of (model, config, step)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please specify a valid checkpoint path with --checkpoint")
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
        sys.exit(1)

    # Load config from meta directory
    run_dir = checkpoint_path.parent.parent.parent
    config_path = run_dir / "meta" / "config.json"

    try:
        import json
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Reconstruct BitNetConfig from JSON
        model_config = config_dict["model"]
        config = BitNetConfig(
            vocab_size=model_config["vocab_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            num_kv_heads=model_config["num_kv_heads"],
            ffn_hidden_size=model_config["ffn_hidden_size"],
            max_seq_length=model_config["max_seq_length"],
            norm_eps=model_config["norm_eps"],
        )
    except FileNotFoundError:
        print(f"\nError: Config not found at {config_path}")
        print("Make sure the run directory structure is intact.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading config: {e}")
        sys.exit(1)

    step = checkpoint["step"]

    # Create model and load weights
    model = BitNetModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters (step {step:,})")

    return model, config, step


def setup_interactive_session(
    checkpoint_path: Path, device: torch.device
) -> Tuple[BitNetModel, PreTrainedTokenizer, torch.device]:
    """Setup model, tokenizer, and device for interactive session.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to use

    Returns:
        Tuple of (model, tokenizer, device)
    """
    # Load model
    model, config, step = load_checkpoint(checkpoint_path, device)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    print()

    return model, tokenizer, device


def generate_text(
    prompt: str,
    model: BitNetModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_length: int = 100,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> str:
    """Generate text continuation for a prompt.

    Args:
        prompt: Input text prompt
        model: BitNetModel instance
        tokenizer: Tokenizer instance
        device: Device to run on
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Generated text (full sequence including prompt)
    """
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Check if prompt is too long
    max_context = model.config.max_seq_length - max_length
    if input_ids.shape[1] > max_context:
        print(
            f"\nWarning: Prompt too long ({input_ids.shape[1]} tokens). "
            f"Truncating to {max_context} tokens.\n"
        )
        input_ids = input_ids[:, -max_context:]

    # Generate
    with torch.no_grad():
        try:
            generated_ids = generate_typical(
                model,
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\nError: Out of memory. Try using --device cpu or shorter prompts.")
                return ""
            raise

    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text


def interactive_loop(
    model: BitNetModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_length: int,
) -> None:
    """Main interactive REPL loop.

    Args:
        model: BitNetModel instance
        tokenizer: Tokenizer instance
        device: Device to run on
        max_length: Maximum tokens to generate per prompt
    """
    print("=" * 80)
    print("Interactive BitNet Chat (Claude/ChatGPT-style generation)")
    print("=" * 80)
    print(f"Settings: temperature=0.9, top_p=0.95, max_length={max_length}")
    print(f"Device: {device}")
    print()
    print("Commands: /quit to exit, /clear for separator")
    print("=" * 80)
    print()

    while True:
        try:
            # Get user input
            prompt = input("You: ").strip()

            # Handle special commands
            if prompt == "/quit":
                print("\nGoodbye!")
                break
            elif prompt == "/clear":
                print("\n" + "=" * 80 + "\n")
                continue
            elif not prompt:
                continue

            # Generate response
            generated_text = generate_text(
                prompt, model, tokenizer, device, max_length
            )

            if generated_text:
                print(f"Model: {generated_text}")
                print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError during generation: {e}")
            print("Try a different prompt or /quit to exit.\n")


def find_latest_checkpoint() -> Path:
    """Find the latest checkpoint in the runs directory.

    Returns:
        Path to latest checkpoint file
    """
    runs_dir = Path("runs")

    if not runs_dir.exists():
        print("Error: No runs directory found.")
        print("Please train a model first or specify --checkpoint")
        sys.exit(1)

    # Find the most recently modified run directory
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not run_dirs:
        print("Error: No run directories found in runs/")
        print("Please train a model first or specify --checkpoint")
        sys.exit(1)

    # Search each run directory for checkpoints
    for run_dir in run_dirs:
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists():
            # Find latest step
            step_dirs = sorted(checkpoints_dir.glob("step_*"), reverse=True)
            if step_dirs:
                checkpoint_file = step_dirs[0] / "checkpoint.pt"
                if checkpoint_file.exists():
                    print(f"Using checkpoint from {run_dir.name}")
                    return checkpoint_file

    print("Error: Could not find any checkpoints in runs/")
    print("Please specify checkpoint path with --checkpoint")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with trained BitNet model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: auto-detect latest)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: cuda or cpu (default: auto-detect)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint()

    # Setup device
    device = torch.device(args.device)

    # Setup session
    model, tokenizer, device = setup_interactive_session(checkpoint_path, device)

    # Start interactive loop
    interactive_loop(model, tokenizer, device, args.max_length)


if __name__ == "__main__":
    main()
