from __future__ import annotations

from typing import Optional, Sequence

import torch


def _shape(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(s) for s in x.shape)


def assert_rank(x: torch.Tensor, rank: int, *, name: str = "tensor") -> None:
    if x.dim() != rank:
        raise ValueError(f"{name} must have rank {rank}, got rank {x.dim()} with shape {_shape(x)}")


def assert_last_dim(x: torch.Tensor, last: int, *, name: str = "tensor") -> None:
    if x.shape[-1] != last:
        raise ValueError(
            f"{name} last dim must be {last}, got {int(x.shape[-1])} with shape {_shape(x)}"
        )


def assert_same_dtype(a: torch.Tensor, b: torch.Tensor, *, a_name="a", b_name="b") -> None:
    if a.dtype != b.dtype:
        raise ValueError(f"{a_name}.dtype ({a.dtype}) != {b_name}.dtype ({b.dtype})")


def assert_same_device(a: torch.Tensor, b: torch.Tensor, *, a_name="a", b_name="b") -> None:
    if a.device != b.device:
        raise ValueError(f"{a_name}.device ({a.device}) != {b_name}.device ({b.device})")


def check_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """
    Expected:
      q: (B, H, Tq, Dh)
      k: (B, H, Tk, Dh)
      v: (B, H, Tk, Dh)
    """
    assert_rank(q, 4, name="q")
    assert_rank(k, 4, name="k")
    assert_rank(v, 4, name="v")

    Bq, Hq, _, Dq = _shape(q)
    Bk, Hk, Tk, Dk = _shape(k)
    Bv, Hv, Tv, Dv = _shape(v)

    if Bq != Bk or Bq != Bv:
        raise ValueError(f"Batch mismatch: q{_shape(q)} k{_shape(k)} v{_shape(v)}")
    if Hq != Hk or Hq != Hv:
        raise ValueError(f"Head mismatch: q{_shape(q)} k{_shape(k)} v{_shape(v)}")
    if Tk != Tv:
        raise ValueError(f"Key/value length mismatch: k{_shape(k)} v{_shape(v)}")
    if Dq != Dk or Dq != Dv:
        raise ValueError(f"Head dim mismatch: q{_shape(q)} k{_shape(k)} v{_shape(v)}")

    assert_same_device(q, k, a_name="q", b_name="k")
    assert_same_device(q, v, a_name="q", b_name="v")


def check_key_padding_mask(
    key_padding_mask: torch.Tensor, *, batch: int, seqlen: int
) -> None:
    """
    Expected keep-mask shape:
      key_padding_mask: (B, S) boolean where True=keep, False=mask
    """
    if key_padding_mask.dtype != torch.bool:
        raise ValueError(f"key_padding_mask must be bool, got {key_padding_mask.dtype}")
    assert_rank(key_padding_mask, 2, name="key_padding_mask")
    B, S = _shape(key_padding_mask)
    if B != batch or S != seqlen:
        raise ValueError(
            f"key_padding_mask must have shape (B,S)=({batch},{seqlen}), got {_shape(key_padding_mask)}"
        )


def check_attn_mask_shape(
    attn_mask: torch.Tensor,
    *,
    B: int,
    H: int,
    Tq: int,
    Tk: int,
) -> None:
    """
    Allows masks broadcastable to (B,H,Tq,Tk).
    Accepted ranks:
      - (Tq, Tk)
      - (B, Tq, Tk)
      - (B, 1, Tq, Tk)
      - (B, H, Tq, Tk)

    dtype:
      - bool keep-mask OR additive float mask
    """
    if attn_mask.dim() not in (2, 3, 4):
        raise ValueError(
            f"attn_mask must have rank 2/3/4, got {attn_mask.dim()} with shape {_shape(attn_mask)}"
        )

    if attn_mask.dtype == torch.bool:
        pass
    elif attn_mask.dtype.is_floating_point:
        pass
    else:
        raise ValueError(
            f"attn_mask dtype must be bool or floating, got {attn_mask.dtype}"
        )

    sh = _shape(attn_mask)

    if attn_mask.dim() == 2:
        if sh != (Tq, Tk):
            raise ValueError(f"attn_mask (2D) must be (Tq,Tk)=({Tq},{Tk}), got {sh}")
        return

    if attn_mask.dim() == 3:
        if sh != (B, Tq, Tk):
            raise ValueError(f"attn_mask (3D) must be (B,Tq,Tk)=({B},{Tq},{Tk}), got {sh}")
        return

    # dim == 4
    b, h, tq, tk = sh
    if b != B or tq != Tq or tk != Tk:
        raise ValueError(
            f"attn_mask (4D) must match (B,*,Tq,Tk)=({B},*,{Tq},{Tk}), got {sh}"
        )
    if h not in (1, H):
        raise ValueError(f"attn_mask head dim must be 1 or H={H}, got {h}")
