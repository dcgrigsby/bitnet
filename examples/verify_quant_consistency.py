"""Verify that quantization behavior is identical in train vs eval mode."""
import torch
from bitnet.linear import BitLinear
from bitnet.quant import activation_quant, weight_quant


def test_bitlinear_consistency():
    """Test that BitLinear applies quantization consistently in train/eval modes."""

    # Create a BitLinear layer with fixed seed
    torch.manual_seed(42)
    layer = BitLinear(in_features=128, out_features=64)

    # Create a test input
    input_tensor = torch.randn(2, 32, 128)

    # Test in training mode
    layer.train()
    with torch.no_grad():
        output_train = layer(input_tensor.clone())

    # Test in eval mode
    layer.eval()
    with torch.no_grad():
        output_eval = layer(input_tensor.clone())

    # Check if outputs are identical (they should be, since quantization is always applied)
    if torch.allclose(output_train, output_eval, atol=1e-6):
        print("✓ BitLinear quantization is consistent between train and eval modes")
    else:
        print("✗ MISMATCH: BitLinear outputs differ between train and eval modes")
        print(f"  Max difference: {(output_train - output_eval).abs().max().item():.2e}")

    return torch.allclose(output_train, output_eval, atol=1e-6)


def test_weight_quantization_deterministic():
    """Test that weight quantization is deterministic."""

    torch.manual_seed(42)
    weights = torch.randn(64, 128)

    # Quantize multiple times
    q1 = weight_quant(weights)
    q2 = weight_quant(weights)

    if torch.allclose(q1, q2):
        print("✓ Weight quantization is deterministic")
    else:
        print("✗ MISMATCH: Weight quantization is non-deterministic")

    return torch.allclose(q1, q2)


def test_activation_quantization_deterministic():
    """Test that activation quantization is deterministic."""

    torch.manual_seed(42)
    activations = torch.randn(2, 32, 128)

    # Quantize multiple times
    q1 = activation_quant(activations)
    q2 = activation_quant(activations)

    if torch.allclose(q1, q2):
        print("✓ Activation quantization is deterministic")
    else:
        print("✗ MISMATCH: Activation quantization is non-deterministic")

    return torch.allclose(q1, q2)


def test_nan_inf_in_logits():
    """Test for NaN/Inf values in model outputs."""
    torch.manual_seed(42)
    from bitnet.config import BitNetConfig
    from bitnet.transformer import BitNetModel

    config = BitNetConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        ffn_hidden_size=128,
    )

    model = BitNetModel(config)
    model.eval()

    # Create a test input
    input_ids = torch.randint(0, 1000, (2, 16))

    with torch.no_grad():
        logits = model(input_ids)

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()

    if not has_nan and not has_inf:
        print("✓ No NaN/Inf values in logits")
    else:
        print(f"✗ MISMATCH: Found NaN={has_nan}, Inf={has_inf} in logits")
        if has_nan:
            nan_count = torch.isnan(logits).sum().item()
            print(f"  NaN count: {nan_count} / {logits.numel()}")
        if has_inf:
            inf_count = torch.isinf(logits).sum().item()
            print(f"  Inf count: {inf_count} / {logits.numel()}")

    return not has_nan and not has_inf


if __name__ == "__main__":
    print("Running quantization consistency checks...\n")

    results = []
    results.append(test_bitlinear_consistency())
    results.append(test_weight_quantization_deterministic())
    results.append(test_activation_quantization_deterministic())
    results.append(test_nan_inf_in_logits())

    print("\n" + "="*50)
    if all(results):
        print("All checks passed ✓")
    else:
        print(f"Some checks failed: {sum(results)}/{len(results)} passed")
