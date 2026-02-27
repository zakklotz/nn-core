# src/nncore/functional/attention.py

import math
from typing import Optional, Callable

import torch
import torch.nn.functional as F

from nncore.utils.shapes import check_qkv, check_attn_mask_shape

def _get_mask_value(dtype: torch.dtype) -> float:
    # Safe large negative for masking
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return torch.finfo(dtype).min


def _make_causal_mask(
    T_q: int,
    T_k: int,
    device: torch.device,
    as_bool: bool = True,
) -> torch.Tensor:
    # True = keep, False = mask
    mask = torch.tril(torch.ones(T_q, T_k, device=device, dtype=torch.bool))
    if as_bool:
        return mask
    return mask.to(torch.float32)


def _combine_masks(
    causal_mask: Optional[torch.Tensor],
    attn_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Returns an additive mask (same dtype as scores) where masked positions are a large negative.
    Accepts:
      - causal_mask: boolean (T_q, T_k) keep-mask
      - attn_mask: boolean keep-mask or additive mask broadcastable to scores
    """
    if causal_mask is None and attn_mask is None:
        return None

    # If attn_mask is already additive, return it (caller will broadcast if needed).
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        return attn_mask

    # Now both causal_mask (if present) and attn_mask (if present) are boolean keep-masks.
    if causal_mask is not None:
        combined = causal_mask if attn_mask is None else (causal_mask & attn_mask)
    else:
        combined = attn_mask  # must be boolean here

    mask_value = _get_mask_value(dtype)
    additive = torch.zeros_like(combined, dtype=dtype)
    additive = additive.masked_fill(~combined, mask_value)
    return additive


def _sdpa_manual(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    dropout_p: float,
    scale: Optional[float],
    normalize: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """
    Manual scaled dot-product attention.

    q, k, v: (B, H, Tq/Tk, Dh)
    attn_mask: boolean keep-mask or additive mask broadcastable to (B, H, Tq, Tk) or (Tq, Tk)
    """
    _, _, T_q, D_h = q.shape
    T_k = k.shape[-2]

    if scale is None:
        scale = 1.0 / math.sqrt(D_h)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T_q, T_k)

    causal_mask = None
    if is_causal:
        causal_mask = _make_causal_mask(T_q, T_k, device=q.device, as_bool=True)

    combined_mask = _combine_masks(causal_mask, attn_mask, dtype=scores.dtype)

    if combined_mask is not None:
        # Broadcast mask to (B, H, T_q, T_k)
        if combined_mask.dim() == 2:
            combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)  # (1,1,Tq,Tk)
        elif combined_mask.dim() == 3:
            combined_mask = combined_mask.unsqueeze(1)  # (B,1,Tq,Tk)
        scores = scores + combined_mask

    if normalize is None:
        attn = torch.softmax(scores, dim=-1)
    else:
        attn = normalize(scores)

    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    out = torch.matmul(attn, v)  # (B, H, T_q, Dh)
    return out


def _sdpa_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    dropout_p: float,
    scale: Optional[float],
    normalize: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """
    PyTorch SDPA backend. Does not support custom normalization.
    """
    if normalize is not None:
        raise ValueError(
            "SDPA backend does not support custom normalize; use backend='manual'."
        )

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    backend: str = "manual",
    normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    q, k, v: (B, H, T, D_h)
    Returns: (B, H, T, D_h)

    normalize:
      A callable that maps scores (B,H,Tq,Tk) -> weights (B,H,Tq,Tk).
      If None, uses softmax(scores, dim=-1).
      Only supported for backend='manual'.
    """
    # Validate q/k/v shapes (B,H,T,Dh) and compatibility
    check_qkv(q, k, v)

    B, H, Tq, _ = q.shape
    Tk = k.shape[-2]

    # Validate attn_mask shapes if provided
    if attn_mask is not None:
        check_attn_mask_shape(attn_mask, B=B, H=H, Tq=Tq, Tk=Tk)

    if backend == "manual":
        return _sdpa_manual(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p,
            scale=scale,
            normalize=normalize,
        )
    elif backend == "sdpa":
        return _sdpa_torch(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p,
            scale=scale,
            normalize=normalize,
        )
    else:
        raise ValueError(f"Unknown attention backend: {backend!r}")
