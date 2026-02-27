import torch

from nncore.functional import attention_forward
from nncore.layers import MultiheadAttention


def test_attention_backends_outputs_close():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)

    y_manual = attention_forward(q, k, v, backend="manual", dropout_p=0.0, is_causal=True)
    y_sdpa = attention_forward(q, k, v, backend="sdpa", dropout_p=0.0, is_causal=True)
    y_auto = attention_forward(q, k, v, backend="auto", dropout_p=0.0, is_causal=True)

    assert y_manual.shape == y_sdpa.shape == y_auto.shape
    assert torch.isfinite(y_manual).all()
    assert torch.isfinite(y_sdpa).all()
    assert torch.isfinite(y_auto).all()
    assert torch.allclose(y_manual, y_sdpa, atol=1e-4, rtol=1e-3)
    assert torch.allclose(y_auto, y_sdpa, atol=1e-4, rtol=1e-3)


def test_multihead_attention_auto_runs():
    torch.manual_seed(0)
    attn = MultiheadAttention(d_model=32, num_heads=4, backend="auto", attn_dropout_p=0.0)
    x = torch.randn(2, 8, 32)
    y = attn(x, is_causal=True)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
