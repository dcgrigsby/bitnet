"""Text generation utilities for BitNet models."""

from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from bitnet.transformer import BitNetModel


def generate_greedy(
    model: BitNetModel,
    input_ids: torch.Tensor,
    max_length: int = 50,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate text using greedy decoding (argmax at each step).

    Args:
        model: BitNetModel instance
        input_ids: Input token IDs of shape [1, seq_len]
        max_length: Maximum number of tokens to generate
        eos_token_id: Stop generation if this token is generated

    Returns:
        Generated token IDs of shape [1, seq_len + generated_len]
    """
    generated = input_ids.clone()

    for _ in range(max_length):
        # Forward pass
        logits = model(generated)

        # Greedy: take argmax
        next_token = torch.argmax(logits[0, -1, :], dim=-1, keepdim=True)

        # Append to sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        # Stop if EOS token generated
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated


def generate_typical(
    model: BitNetModel,
    input_ids: torch.Tensor,
    max_length: int = 50,
    temperature: float = 0.9,
    top_p: float = 0.95,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate text using typical sampling (temperature + nucleus/top-p).

    Args:
        model: BitNetModel instance
        input_ids: Input token IDs of shape [1, seq_len]
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (default: 0.9)
        top_p: Nucleus sampling threshold (default: 0.95)
        eos_token_id: Stop generation if this token is generated

    Returns:
        Generated token IDs of shape [1, seq_len + generated_len]
    """
    generated = input_ids.clone()

    for _ in range(max_length):
        # Forward pass
        logits = model(generated)

        # Apply temperature
        logits = logits[0, -1, :] / temperature

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff
        mask = cumsum_probs <= top_p
        mask[0] = True  # Always include top token

        # Filter and renormalize
        sorted_probs = sorted_probs[mask]
        sorted_indices = sorted_indices[mask]
        probs_normalized = sorted_probs / sorted_probs.sum()

        # Sample
        idx = torch.multinomial(probs_normalized, 1)
        next_token = sorted_indices[idx]

        # Append to sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        # Stop if EOS token generated
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated


def generate_stress(
    model: BitNetModel,
    input_ids: torch.Tensor,
    max_length: int = 50,
    temperature: float = 1.5,
    top_k: int = 10,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate text using high-temperature + top-k (stress test).

    This mode tests model confidence by using high temperature and
    restricting to top-k tokens only.

    Args:
        model: BitNetModel instance
        input_ids: Input token IDs of shape [1, seq_len]
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (default: 1.5, high)
        top_k: Number of top tokens to consider (default: 10)
        eos_token_id: Stop generation if this token is generated

    Returns:
        Generated token IDs of shape [1, seq_len + generated_len]
    """
    generated = input_ids.clone()

    for _ in range(max_length):
        # Forward pass
        logits = model(generated)

        # Apply temperature
        logits = logits[0, -1, :] / temperature

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Top-k filtering
        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
        probs_normalized = top_probs / top_probs.sum()

        # Sample
        idx = torch.multinomial(probs_normalized, 1)
        next_token = top_indices[idx]

        # Append to sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        # Stop if EOS token generated
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated


def compute_generation_metrics(
    generated_ids: torch.Tensor,
    model: BitNetModel,
    input_len: int,
) -> dict[str, Any]:
    """Compute metrics for generated sequence.

    Args:
        generated_ids: Generated token IDs of shape [1, total_len]
        model: BitNetModel instance
        input_len: Length of input prompt (to exclude from metrics)

    Returns:
        Dictionary with metrics:
            - mean_entropy: Average entropy across generated positions
            - max_repetition: Maximum times any token repeats
            - unique_tokens: Number of unique tokens generated
            - total_tokens: Total tokens generated (excluding prompt)
    """
    # Get generated portion only
    gen_tokens = generated_ids[0, input_len:].tolist()

    if not gen_tokens:
        return {
            "mean_entropy": 0.0,
            "max_repetition": 0,
            "unique_tokens": 0,
            "total_tokens": 0,
        }

    # Compute entropies
    entropies = []
    with torch.no_grad():
        for pos in range(input_len, generated_ids.shape[1]):
            logits = model(generated_ids[:, :pos])
            probs = F.softmax(logits[0, -1, :], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            entropies.append(entropy)

    # Count repetitions
    from collections import Counter

    token_counts = Counter(gen_tokens)
    max_repetition = max(token_counts.values()) if token_counts else 0
    unique_tokens = len(token_counts)

    return {
        "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "max_repetition": max_repetition,
        "unique_tokens": unique_tokens,
        "total_tokens": len(gen_tokens),
    }
