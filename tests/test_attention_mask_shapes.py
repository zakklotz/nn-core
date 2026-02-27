import pytest
import torch

from nncore.utils.shapes import check_attn_mask_shape, check_qkv


def test_check_qkv_ok():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 9, 16)
    v = torch.randn(2, 4, 9, 16)
    check_qkv(q, k, v)


def test_check_qkv_bad_heads():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 3, 9, 16)
    v = torch.randn(2, 3, 9, 16)
    with pytest.raises(ValueError):
        check_qkv(q, k, v)


def test_check_attn_mask_shapes_ok():
    B, H, Tq, Tk = 2, 4, 8, 9

    m2 = torch.ones(Tq, Tk, dtype=torch.bool)
    check_attn_mask_shape(m2, B=B, H=H, Tq=Tq, Tk=Tk)

    m3 = torch.ones(B, Tq, Tk, dtype=torch.bool)
    check_attn_mask_shape(m3, B=B, H=H, Tq=Tq, Tk=Tk)

    m4a = torch.zeros(B, 1, Tq, Tk)  # additive float mask
    check_attn_mask_shape(m4a, B=B, H=H, Tq=Tq, Tk=Tk)

    m4b = torch.zeros(B, H, Tq, Tk)
    check_attn_mask_shape(m4b, B=B, H=H, Tq=Tq, Tk=Tk)


def test_check_attn_mask_shapes_bad():
    B, H, Tq, Tk = 2, 4, 8, 9
    bad = torch.ones(B, H, Tk, Tq, dtype=torch.bool)  # swapped
    with pytest.raises(ValueError):
        check_attn_mask_shape(bad, B=B, H=H, Tq=Tq, Tk=Tk)
