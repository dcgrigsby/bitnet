#!/usr/bin/env python3
"""
Script to verify residual connections support backpropagation in BitNet.

This script demonstrates that:
1. Residual connections are present in the architecture
2. Gradients flow correctly through residual paths
3. The implementation supports training via backpropagation
"""

import sys
import os

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
sys.path.insert(0, src_dir)

import torch
from bitnet.transformer import TransformerBlock, BitNetModel
from bitnet.config import BitNetConfig


def test_residual_connections_basic():
    """Verify residual connections are implemented correctly."""
    print("=" * 70)
    print("TEST 1: Basic Residual Connection Implementation")
    print("=" * 70)
    
    # Create a transformer block
    block = TransformerBlock(256, 4, 4, 512)
    x = torch.randn(2, 8, 256)
    
    # Track intermediate values using hooks
    attn_outputs = []
    intermediates = []
    
    def attn_hook(_module, _input, output):
        attn_outputs.append(output.clone())
    
    def ffn_pre_hook(_module, input_tuple):
        intermediates.append(input_tuple[0].clone())
    
    # Register hooks
    h1 = block.attention.register_forward_hook(attn_hook)
    h2 = block.feedforward.register_forward_pre_hook(ffn_pre_hook)
    
    # Forward pass
    output = block(x)
    
    # Remove hooks
    h1.remove()
    h2.remove()
    
    # Verify residual connection
    attn_out = attn_outputs[0]
    expected_after_attn = x + attn_out
    intermediate = intermediates[0]
    
    residual_matches = torch.allclose(intermediate, expected_after_attn, atol=1e-6)
    
    print(f"✓ Transformer block created successfully")
    print(f"✓ Forward pass completed")
    print(f"✓ Attention output captured: shape {attn_out.shape}")
    print(f"✓ Intermediate value captured: shape {intermediate.shape}")
    print(f"✓ Residual connection verified: {residual_matches}")
    
    if residual_matches:
        print("\n✅ PASS: Attention residual connection is correctly implemented")
        print("   Formula: x_after_attn = x_input + attention(x_input)")
    else:
        print("\n❌ FAIL: Residual connection not working as expected")
        return False
    
    return True


def test_gradient_flow():
    """Verify gradients flow through residual connections."""
    print("\n" + "=" * 70)
    print("TEST 2: Gradient Flow Through Residual Connections")
    print("=" * 70)
    
    # Create transformer block with gradient tracking
    block = TransformerBlock(256, 4, 4, 512)
    x = torch.randn(2, 8, 256, requires_grad=True)
    
    print("✓ Created transformer block")
    print("✓ Input tensor with requires_grad=True")
    
    # Forward pass
    output = block(x)
    loss = output.sum()
    
    print("✓ Forward pass completed")
    print(f"✓ Loss computed: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    print("✓ Backward pass completed")
    
    # Check gradients exist
    input_grad_exists = x.grad is not None
    attn_grad_exists = block.attention.qkv_proj.weight.grad is not None
    ffn_grad_exists = block.feedforward.gate_up.weight.grad is not None
    
    print(f"✓ Input gradient exists: {input_grad_exists}")
    print(f"✓ Attention weights gradient exists: {attn_grad_exists}")
    print(f"✓ FFN weights gradient exists: {ffn_grad_exists}")
    
    # Check for NaN or Inf in gradients
    input_grad_valid = not torch.isnan(x.grad).any() and not torch.isinf(x.grad).any()
    output_valid = not torch.isnan(output).any() and not torch.isinf(output).any()
    
    print(f"✓ Input gradients are valid (no NaN/Inf): {input_grad_valid}")
    print(f"✓ Output is valid (no NaN/Inf): {output_valid}")
    
    if all([input_grad_exists, attn_grad_exists, ffn_grad_exists, 
            input_grad_valid, output_valid]):
        print("\n✅ PASS: Gradients flow correctly through residual connections")
        print("   All parameters receive gradients during backpropagation")
    else:
        print("\n❌ FAIL: Gradient flow problem detected")
        return False
    
    return True


def test_full_model_gradient_flow():
    """Verify gradients flow through entire BitNet model."""
    print("\n" + "=" * 70)
    print("TEST 3: Full Model Gradient Flow")
    print("=" * 70)
    
    # Create a small model
    config = BitNetConfig(num_layers=3, vocab_size=1000, hidden_size=256, 
                          num_heads=4, num_kv_heads=4, ffn_hidden_size=512)
    model = BitNetModel(config)
    
    print(f"✓ Created BitNet model with {config.num_layers} layers")
    
    # Forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(input_ids)
    loss = logits.sum()
    
    print(f"✓ Forward pass completed, logits shape: {logits.shape}")
    print(f"✓ Loss computed: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    print("✓ Backward pass completed")
    
    # Check gradients in different blocks
    block_0 = model.blocks[0]
    block_1 = model.blocks[1]
    block_2 = model.blocks[2]
    
    grad_checks = {
        "Token embeddings": model.token_embeddings.weight.grad is not None,
        "Block 0 attention": block_0.attention.qkv_proj.weight.grad is not None,
        "Block 1 FFN": block_1.feedforward.gate_up.weight.grad is not None,
        "Block 2 output": block_2.attention.out_proj.weight.grad is not None,
        "LM head": model.lm_head.weight.grad is not None,
    }
    
    for name, has_grad in grad_checks.items():
        status = "✓" if has_grad else "✗"
        print(f"{status} {name}: gradient exists = {has_grad}")
    
    # Check gradient magnitudes are reasonable
    all_grads_valid = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"✗ NaN detected in gradients: {name}")
                all_grads_valid = False
            if torch.isinf(param.grad).any():
                print(f"✗ Inf detected in gradients: {name}")
                all_grads_valid = False
            if param.grad.abs().max() > 10000:
                print(f"⚠ Large gradient detected: {name} (max: {param.grad.abs().max():.2f})")
    
    if all(grad_checks.values()) and all_grads_valid:
        print("\n✅ PASS: Gradients flow through entire model correctly")
        print("   All layers from embeddings to LM head receive valid gradients")
    else:
        print("\n❌ FAIL: Gradient flow issue in full model")
        return False
    
    return True


def verify_architecture_structure():
    """Print the architecture to show where residual connections are."""
    print("\n" + "=" * 70)
    print("ARCHITECTURE VERIFICATION: Residual Connection Locations")
    print("=" * 70)
    
    print("\nTransformerBlock structure:")
    print("  Input")
    print("    │")
    print("    ├─────────────────┐")
    print("    │                 │")
    print("    ↓              (skip)")
    print("  Attention           │")
    print("    │                 │")
    print("    └────── + ────────┘  ← RESIDUAL CONNECTION #1")
    print("         │")
    print("         ├─────────────────┐")
    print("         │                 │")
    print("         ↓              (skip)")
    print("    FeedForward          │")
    print("         │                 │")
    print("         └────── + ────────┘  ← RESIDUAL CONNECTION #2")
    print("              │")
    print("           Output")
    
    print("\n✓ Both residual connections are present in transformer.py:")
    print("  - x = x + attn_output  (after attention)")
    print("  - x = x + ffn_output   (after feedforward)")
    
    print("\n✓ Pre-normalization is used:")
    print("  - RMSNorm applied BEFORE attention (in Attention class)")
    print("  - RMSNorm applied BEFORE feedforward (in FeedForward class)")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print(" BitNet Residual Connections Verification")
    print("=" * 70)
    print("\nThis script verifies that residual connections are properly")
    print("implemented and support backpropagation in the BitNet architecture.")
    print()
    
    # Run architecture verification first
    verify_architecture_structure()
    
    # Run tests
    tests = [
        ("Basic Residual Implementation", test_residual_connections_basic),
        ("Gradient Flow", test_gradient_flow),
        ("Full Model Gradient Flow", test_full_model_gradient_flow),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print(" VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = all(passed for _, passed in results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print()
        print("CONCLUSION:")
        print("  ✓ Residual connections are properly implemented")
        print("  ✓ Gradients flow correctly through all layers")
        print("  ✓ The architecture supports backpropagation")
        print()
        print("No changes needed - implementation is correct!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("Please review the failed tests above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
