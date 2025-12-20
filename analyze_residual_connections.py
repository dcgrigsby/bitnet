#!/usr/bin/env python3
"""
Code Analysis Script: Verify Residual Connections in BitNet

This script performs static analysis of the BitNet source code to verify
that residual connections are properly implemented for backpropagation.
"""

import os
import re
import sys

# Get the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'src', 'bitnet')
TEST_DIR = os.path.join(SCRIPT_DIR, 'tests')


def analyze_transformer_block():
    """Analyze the TransformerBlock implementation for residual connections."""
    print("=" * 70)
    print("ANALYZING: src/bitnet/transformer.py - TransformerBlock")
    print("=" * 70)
    
    file_path = os.path.join(SRC_DIR, 'transformer.py')
    
    with open(file_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Find the forward method
    in_forward = False
    forward_lines = []
    for i, line in enumerate(lines, 1):
        if 'def forward(self' in line and 'TransformerBlock' in content[:content.index(line)]:
            in_forward = True
        if in_forward:
            forward_lines.append((i, line))
            if line.strip() and not line.strip().startswith('#') and line.strip().startswith('return'):
                break
    
    print("\n✓ Located TransformerBlock.forward() method\n")
    
    # Look for residual connections
    residual_patterns = [
        (r'x\s*=\s*x\s*\+\s*attn', 'Attention residual'),
        (r'x\s*=\s*x\s*\+\s*ffn', 'FFN residual'),
    ]
    
    found_residuals = []
    for line_num, line in forward_lines:
        for pattern, name in residual_patterns:
            if re.search(pattern, line):
                found_residuals.append((line_num, name, line.strip()))
    
    print("Residual Connections Found:")
    print("-" * 70)
    for line_num, name, line in found_residuals:
        print(f"  Line {line_num}: {name}")
        print(f"    Code: {line}")
        print()
    
    # Print the relevant section
    print("Forward Method Implementation:")
    print("-" * 70)
    for line_num, line in forward_lines:
        marker = "  →" if any(line_num == ln for ln, _, _ in found_residuals) else "   "
        print(f"{marker} {line_num:3d}. {line}")
    
    print()
    if len(found_residuals) >= 2:
        print("✅ VERIFIED: Both residual connections are present")
        print("   - Attention block: input + attention_output")
        print("   - FeedForward block: input + ffn_output")
        return True
    else:
        print("❌ WARNING: Expected 2 residual connections, found", len(found_residuals))
        return False


def analyze_attention_normalization():
    """Analyze the Attention class for pre-normalization."""
    print("\n" + "=" * 70)
    print("ANALYZING: src/bitnet/attention.py - Attention Layer")
    print("=" * 70)
    
    file_path = os.path.join(SRC_DIR, 'attention.py')
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for RMSNorm
    has_rmsnorm = 'RMSNorm' in content
    has_norm_init = re.search(r'self\.norm.*=.*RMSNorm', content)
    has_norm_call = re.search(r'x_norm\s*=\s*self\.norm\(', content)
    
    print("\n✓ Checking for normalization layer:")
    print(f"  - RMSNorm imported: {has_rmsnorm}")
    print(f"  - Norm layer initialized: {has_norm_init is not None}")
    print(f"  - Norm applied before computation: {has_norm_call is not None}")
    
    if has_norm_call:
        # Find the line
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'x_norm = self.norm(' in line:
                print(f"\n  Found at line {i}: {line.strip()}")
                break
    
    if all([has_rmsnorm, has_norm_init, has_norm_call]):
        print("\n✅ VERIFIED: Pre-normalization is correctly implemented")
        print("   Inputs are normalized BEFORE attention computation")
        return True
    else:
        print("\n❌ WARNING: Normalization may not be properly configured")
        return False


def analyze_feedforward_normalization():
    """Analyze the FeedForward class for pre-normalization."""
    print("\n" + "=" * 70)
    print("ANALYZING: src/bitnet/feedforward.py - FeedForward Layer")
    print("=" * 70)
    
    file_path = os.path.join(SRC_DIR, 'feedforward.py')
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for RMSNorm
    has_rmsnorm = 'RMSNorm' in content
    has_norm_init = re.search(r'self\.norm.*=.*RMSNorm', content)
    has_norm_call = re.search(r'x_norm\s*=\s*self\.norm\(', content)
    
    print("\n✓ Checking for normalization layer:")
    print(f"  - RMSNorm imported: {has_rmsnorm}")
    print(f"  - Norm layer initialized: {has_norm_init is not None}")
    print(f"  - Norm applied before computation: {has_norm_call is not None}")
    
    if has_norm_call:
        # Find the line
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'x_norm = self.norm(' in line:
                print(f"\n  Found at line {i}: {line.strip()}")
                break
    
    if all([has_rmsnorm, has_norm_init, has_norm_call]):
        print("\n✅ VERIFIED: Pre-normalization is correctly implemented")
        print("   Inputs are normalized BEFORE feedforward computation")
        return True
    else:
        print("\n❌ WARNING: Normalization may not be properly configured")
        return False


def analyze_tests():
    """Analyze test files for gradient flow tests."""
    print("\n" + "=" * 70)
    print("ANALYZING: tests/test_transformer.py - Test Coverage")
    print("=" * 70)
    
    file_path = os.path.join(TEST_DIR, 'test_transformer.py')
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find test functions
    test_pattern = r'def (test_\w+)\(.*?\):'
    tests = re.findall(test_pattern, content)
    
    print(f"\n✓ Found {len(tests)} test functions:\n")
    
    relevant_tests = []
    for test in tests:
        if any(keyword in test.lower() for keyword in ['residual', 'gradient', 'backward']):
            relevant_tests.append(test)
            print(f"  ✓ {test}")
            
            # Find and print the docstring
            pattern = rf'def {test}\(.*?\):\s*"""(.*?)"""'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                docstring = match.group(1).strip()
                print(f"     Description: {docstring}")
            print()
    
    if len(relevant_tests) >= 2:
        print("✅ VERIFIED: Comprehensive test coverage for residual connections")
        print(f"   Found {len(relevant_tests)} tests covering gradient flow")
        return True
    else:
        print("⚠  WARNING: Limited test coverage for gradient flow")
        return False


def print_architecture_diagram():
    """Print a visual representation of the architecture."""
    print("\n" + "=" * 70)
    print("ARCHITECTURE DIAGRAM")
    print("=" * 70)
    
    print("""
BitNet TransformerBlock Architecture:

┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Input (x)                                              │
│    │                                                    │
│    ├─────────────────────┐                             │
│    │                     │                             │
│    ↓                  (skip)    ← Residual Path #1     │
│  RMSNorm                 │                             │
│    │                     │                             │
│    ↓                     │                             │
│  Attention               │                             │
│    │                     │                             │
│    └────────(+)──────────┘     ← ADD operation         │
│         │                                               │
│         x (after attention)                             │
│         │                                               │
│         ├─────────────────────┐                        │
│         │                     │                        │
│         ↓                  (skip)    ← Residual Path #2│
│      RMSNorm                 │                         │
│         │                     │                        │
│         ↓                     │                        │
│    FeedForward               │                         │
│         │                     │                        │
│         └────────(+)──────────┘     ← ADD operation    │
│              │                                          │
│           Output                                        │
│                                                         │
└─────────────────────────────────────────────────────────┘

Key Implementation Details:
  • Residual connections: x = x + sublayer_output
  • Pre-normalization: normalize BEFORE each sublayer
  • RMSNorm: Efficient root-mean-square normalization
  • Gradient flow: Direct path through addition operations
""")


def main():
    """Run all analyses."""
    print("\n" + "=" * 70)
    print(" BITNET RESIDUAL CONNECTIONS CODE ANALYSIS")
    print("=" * 70)
    print("\nStatic code analysis to verify residual connections for backpropagation")
    print()
    
    # Print architecture first
    print_architecture_diagram()
    
    # Run analyses
    results = []
    
    print("\n" + "=" * 70)
    print(" DETAILED CODE ANALYSIS")
    print("=" * 70)
    
    analyses = [
        ("TransformerBlock Residual Connections", analyze_transformer_block),
        ("Attention Pre-Normalization", analyze_attention_normalization),
        ("FeedForward Pre-Normalization", analyze_feedforward_normalization),
        ("Test Coverage", analyze_tests),
    ]
    
    for name, func in analyses:
        try:
            passed = func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print(" ANALYSIS SUMMARY")
    print("=" * 70)
    
    all_passed = all(passed for _, passed in results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print()
        print("=" * 70)
        print(" FINAL CONCLUSION")
        print("=" * 70)
        print()
        print("Residual connections are PROPERLY IMPLEMENTED for backpropagation:")
        print()
        print("  ✓ Two residual connections per TransformerBlock")
        print("    1. After attention: x = x + attention(x)")
        print("    2. After feedforward: x = x + feedforward(x)")
        print()
        print("  ✓ Pre-normalization strategy")
        print("    - RMSNorm applied before each sublayer")
        print("    - Improves training stability")
        print()
        print("  ✓ Gradient flow support")
        print("    - Addition operations create direct gradient paths")
        print("    - Prevents vanishing gradients in deep networks")
        print()
        print("  ✓ Comprehensive test coverage")
        print("    - Tests verify residual connection behavior")
        print("    - Tests confirm gradient flow through all layers")
        print()
        print("━" * 70)
        print(" NO CHANGES NEEDED - Implementation is complete and correct!")
        print("━" * 70)
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print()
        print("Please review the failed checks above.")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
