#!/usr/bin/env python3
"""
Train a custom 2K vocabulary tokenizer on TinyStories dataset.

This creates a compact tokenizer suitable for the small model,
reducing vocabulary from 32K (LLaMA) to 2K while maintaining
good coverage of the simple language in TinyStories.
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TinyStories tokenizer"
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2048,
        help="Vocabulary size (default: 2048)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tinystories_tokenizer.json",
        help="Output path for tokenizer (default: tinystories_tokenizer.json)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100000,
        help="Number of stories to use for training (default: 100000)",
    )

    return parser.parse_args()


def train_tokenizer(
    vocab_size: int = 2048,
    num_samples: int = 100000,
) -> Tokenizer:
    """Train BPE tokenizer on TinyStories.

    Args:
        vocab_size: Target vocabulary size
        num_samples: Number of stories to use for training

    Returns:
        Trained tokenizer
    """
    print("Loading TinyStories dataset...")
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    print(f"Training tokenizer with vocab_size={vocab_size}...")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
        show_progress=True,
    )

    # Create text iterator
    def text_iterator():
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            yield item["text"]

    # Train tokenizer
    tokenizer.train_from_iterator(text_iterator(), trainer)

    print("Tokenizer training complete!")
    return tokenizer


def test_tokenizer(tokenizer: Tokenizer) -> None:
    """Test tokenizer on sample text.

    Args:
        tokenizer: Trained tokenizer to test
    """
    print("\n" + "=" * 80)
    print("Testing tokenizer...")
    print("=" * 80)

    test_texts = [
        "Once upon a time, there was a little girl named Lily.",
        "She loved to play with her toys.",
        "One day, she found a big red ball.",
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)

        print(f"\nOriginal: {text}")
        print(f"Tokens:   {len(encoded.ids)} tokens")
        print(f"IDs:      {encoded.ids[:20]}{'...' if len(encoded.ids) > 20 else ''}")
        print(f"Decoded:  {decoded}")


def main() -> None:
    """Main function."""
    args = parse_args()

    print("=" * 80)
    print("TinyStories Tokenizer Training")
    print("=" * 80)
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Training samples: {args.num_samples}")
    print(f"Output path: {args.output}")
    print("=" * 80)
    print()

    # Train tokenizer
    tokenizer = train_tokenizer(
        vocab_size=args.vocab_size,
        num_samples=args.num_samples,
    )

    # Test tokenizer
    test_tokenizer(tokenizer)

    # Save tokenizer
    output_path = Path(args.output)
    tokenizer.save(str(output_path))
    print(f"\n✓ Tokenizer saved to {output_path}")

    # Print vocabulary info
    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: <pad>, <s>, </s>, <unk>")
    print()
    print("=" * 80)
    print("Tokenizer ready for training!")
    print("=" * 80)


if __name__ == "__main__":
    main()
