import pytest
import torch

from bitnet.config import BitNetConfig
from bitnet.transformer import BitNetModel, TransformerBlock


def test_transformer_block_residual_connections():
    """Test transformer block uses residual connections."""

    block = TransformerBlock(256, 4, 4, 512)
    x = torch.randn(2, 8, 256)

    # Capture attention output
    attn_outputs = []

    def attn_hook(module, input, output):
        attn_outputs.append(output)

    handle = block.attention.register_forward_hook(attn_hook)

    # Capture intermediate (after attn+residual, before FFN)
    intermediate = []

    def ffn_pre_hook(module, input):
        intermediate.append(input[0])  # input is a tuple

    ffn_handle = block.feedforward.register_forward_pre_hook(ffn_pre_hook)

    y = block(x)
    handle.remove()
    ffn_handle.remove()

    # Check attention residual: intermediate should equal x + attn_out
    attn_out = attn_outputs[0]
    expected_after_attn = x + attn_out
    assert torch.allclose(intermediate[0], expected_after_attn, atol=1e-6)


def test_transformer_block_attention_then_ffn():
    """Test transformer block applies attention before feedforward."""

    block = TransformerBlock(256, 4, 4, 512)
    x = torch.randn(2, 8, 256)

    # Hooks to track execution order
    execution_order = []

    def attn_hook(module, input, output):
        execution_order.append("attention")

    def ffn_hook(module, input, output):
        execution_order.append("feedforward")

    handle_attn = block.attention.register_forward_hook(attn_hook)
    handle_ffn = block.feedforward.register_forward_hook(ffn_hook)

    y = block(x)

    handle_attn.remove()
    handle_ffn.remove()

    # Verify execution order
    assert execution_order == ["attention", "feedforward"]


def test_transformer_block_forward_backward():
    """Test transformer block supports gradients."""

    hidden_size = 768
    num_heads = 12
    num_kv_heads = 12
    ffn_hidden_size = 3072

    block = TransformerBlock(hidden_size, num_heads, num_kv_heads, ffn_hidden_size)
    x = torch.randn(2, 10, hidden_size, requires_grad=True)
    y = block(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert block.attention.qkv_proj.weight.grad is not None
    assert block.feedforward.gate_up.weight.grad is not None
    assert not torch.isnan(y).any()


def test_bitnet_model_applies_final_norm():
    """Test model applies final RMSNorm before output projection."""

    config = BitNetConfig(num_layers=2)
    model = BitNetModel(config)

    # Hook to verify final norm is applied
    norm_outputs = []

    def norm_hook(module, input, output):
        norm_outputs.append(output)

    handle = model.final_norm.register_forward_hook(norm_hook)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(input_ids)

    handle.remove()

    # Verify final norm was called
    assert len(norm_outputs) > 0
    norm_out = norm_outputs[0]

    # Norm output should be normalized (RMS ≈ 1)
    rms = torch.sqrt((norm_out**2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


def test_bitnet_model_forward_backward():
    """Test BitNetModel supports training."""

    config = BitNetConfig()
    model = BitNetModel(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    # Check embeddings have gradients
    assert model.token_embeddings.weight.grad is not None

    # Check some block parameters have gradients
    assert model.blocks[0].attention.qkv_proj.weight.grad is not None
    assert model.blocks[0].feedforward.gate_up.weight.grad is not None

    # Check output head has gradients
    assert model.lm_head.weight.grad is not None


def test_bitnet_model_all_blocks_executed():
    """Test that all transformer blocks are executed during forward pass."""
    config = BitNetConfig(num_layers=4)
    model = BitNetModel(config)

    # Track which blocks were executed
    blocks_executed = []

    for i, block in enumerate(model.blocks):

        def make_hook(idx):
            def hook(module, input, output):
                blocks_executed.append(idx)

            return hook

        block.register_forward_hook(make_hook(i))

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(input_ids)

    # All blocks should have been executed in order
    assert blocks_executed == [0, 1, 2, 3]


def test_bitnet_model_gradient_flow():
    """Test gradients flow through entire model."""
    config = BitNetConfig(num_layers=3)
    model = BitNetModel(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    # Check gradients in first, middle, and last blocks
    assert model.blocks[0].attention.qkv_proj.weight.grad is not None
    assert model.blocks[1].feedforward.gate_up.weight.grad is not None
    assert model.blocks[2].attention.out_proj.weight.grad is not None

    # Gradients should be bounded
    for param in model.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()
            assert param.grad.abs().max() < 10000


def test_bitnet_model_deterministic():
    """Test model produces deterministic output for same input."""
    config = BitNetConfig(num_layers=2)
    model = BitNetModel(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 10))

    with torch.no_grad():
        logits1 = model(input_ids)
        logits2 = model(input_ids)

    assert torch.allclose(logits1, logits2)
