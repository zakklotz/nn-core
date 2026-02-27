# tests/test_attention_forward.py

import torch

from nncore.layers import MultiheadAttention


def _run_forward(backend: str):
    torch.manual_seed(0)

    B, T, C = 2, 8, 32
    H = 4

    attn = MultiheadAttention(
        d_model=C,
        num_heads=H,
        backend=backend,
        attn_dropout_p=0.0,
        out_dropout_p=0.0,
    )

    x = torch.randn(B, T, C)
    y = attn(x, is_causal=True)

    assert y.shape == (B, T, C)
    assert torch.isfinite(y).all()


def test_attention_forward_manual():
    _run_forward("manual")


def test_attention_forward_sdpa():
    _run_forward("sdpa")


def test_attention_custom_normalize_requires_manual():
    # Custom normalize should error on SDPA backend
    attn = MultiheadAttention(
        d_model=32,
        num_heads=4,
        backend="sdpa",
        normalize=lambda s: torch.softmax(s, dim=-1),
    )
    x = torch.randn(2, 8, 32)

    try:
        _ = attn(x, is_causal=True)
        assert False, "Expected ValueError for custom normalize with SDPA backend"
    except ValueError:
        pass
