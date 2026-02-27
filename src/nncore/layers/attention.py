# src/nncore/layers/attention.py

import torch
import torch.nn as nn

from nncore.functional import attention_forward
from nncore.positional import Rope
from nncore.utils.shapes import check_key_padding_mask

class MultiheadAttention(nn.Module):
    """
    Multi-head attention module.

    - Projects inputs into q/k/v
    - Splits into heads
    - Calls nncore.functional.scaled_dot_product_attention
    - Merges heads and applies output projection

    Inputs:
      x: (B, T, d_model) for self-attention
      context: optional (B, S, d_model) for cross-attention (k/v come from context)

    Masks:
      attn_mask: optional bool keep-mask or additive mask broadcastable to (B, H, T, S) or (T, S)
      key_padding_mask: optional bool keep-mask (B, S) where True means "keep"

    Normalization:
      normalize: optional callable mapping scores (B,H,T,S) -> weights (B,H,T,S).
        - If provided, requires backend="manual" (SDPA backend cannot swap softmax).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        bias: bool = True,
        attn_dropout_p: float = 0.0,
        out_dropout_p: float = 0.0,
        backend: str = "manual",   # "manual", "sdpa", or "auto"
        scale: float | None = None,
        normalize=None,            # callable(scores)->weights, only for manual
        positional: str = "absolute",
        max_seq_len: int = 2048,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = None

        self.attn_dropout_p = float(attn_dropout_p)
        self.out_dropout_p = float(out_dropout_p)
        self.backend = backend
        self.scale = scale  # optional override for 1/sqrt(d_head)
        self.normalize = normalize

        if positional == "rope":
            self.rope = Rope(dim=self.d_head, max_seq_len=max_seq_len)

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.out_dropout = (
            nn.Dropout(self.out_dropout_p) if self.out_dropout_p > 0.0 else nn.Identity()
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) -> (B, H, T, d_head)
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d_head) -> (B, T, d_model)
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        """
        Returns:
          y: (B, T, d_model)
        """
        if context is None:
            context = x

        # Project
        q = self.q_proj(x)         # (B, T, d_model)
        k = self.k_proj(context)   # (B, S, d_model)
        v = self.v_proj(context)   # (B, S, d_model)

        # Validate key_padding_mask if provided: (B,S) bool keep-mask
        if key_padding_mask is not None:
            B = context.shape[0]
            S = context.shape[1]
            check_key_padding_mask(key_padding_mask, batch=B, seqlen=S)

        # Heads
        q = self._split_heads(q)   # (B, H, T, d_head)
        k = self._split_heads(k)   # (B, H, S, d_head)
        v = self._split_heads(v)   # (B, H, S, d_head)

        if self.rope is not None and q.shape[-2] == k.shape[-2]:
            q, k = self.rope.apply(q, k, pos_offset=pos_offset)

        # key_padding_mask (B, S) keep-mask -> broadcastable keep-mask (B,1,1,S)
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be a boolean tensor (True=keep, False=mask).")
            pad = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,S)

            if attn_mask is None:
                attn_mask = pad
            else:
                # If both are boolean keep-masks: AND. If attn_mask is additive, add additive pad.
                if attn_mask.dtype == torch.bool:
                    attn_mask = attn_mask & pad
                else:
                    mask_value = -1e4 if q.dtype in (torch.float16, torch.bfloat16) else torch.finfo(q.dtype).min
                    additive_pad = torch.zeros_like(pad, dtype=q.dtype).masked_fill(~pad, mask_value)
                    attn_mask = attn_mask + additive_pad

        # Attention kernel
        y = attention_forward(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            scale=self.scale,
            backend=self.backend,
            normalize=self.normalize,
        )  # (B, H, T, d_head)

        # Merge + output projection
        y = self._merge_heads(y)     # (B, T, d_model)
        y = self.out_proj(y)         # (B, T, d_model)
        y = self.out_dropout(y)
        return y
