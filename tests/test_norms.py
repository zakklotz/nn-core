import torch

from nncore.blocks.transformer import TransformerBlock
from nncore.layers.norm import RMSNorm


def test_rmsnorm_forward_shape():
    norm = RMSNorm(d_model=16)
    x = torch.randn(2, 5, 16)
    y = norm(x)
    assert y.shape == x.shape


def test_rmsnorm_grad_flow():
    norm = RMSNorm(d_model=8)
    x = torch.randn(3, 4, 8, requires_grad=True)
    y = norm(x).sum()
    y.backward()
    assert norm.weight.grad is not None


def test_transformer_block_with_rmsnorm_runs():
    block = TransformerBlock(
        d_model=16,
        num_heads=4,
        norm="rmsnorm",
        norm_eps=1e-5,
    )
    x = torch.randn(2, 6, 16)
    y = block(x, is_causal=False)
    assert y.shape == x.shape
