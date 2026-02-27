import pytest
import torch

from nncore.functional import scaled_dot_product_attention


def test_attention_wired_checks_bad_mask_raises():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 9, 16)
    v = torch.randn(2, 4, 9, 16)

    # Wrong shape: (B,H,Tk,Tq) instead of (B,H,Tq,Tk)
    bad_mask = torch.ones(2, 4, 9, 8, dtype=torch.bool)

    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, attn_mask=bad_mask, backend="manual")


def test_attention_wired_checks_bad_qkv_raises():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 3, 9, 16)  # mismatched heads
    v = torch.randn(2, 3, 9, 16)

    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v, backend="manual")
