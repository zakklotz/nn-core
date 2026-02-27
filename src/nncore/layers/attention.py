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
    - Calls nncore.functional.attention_forward
    - Merges heads and applies output projection
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
        use_kv_cache: bool = False,
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
        self.scale = scale
        self.normalize = normalize
        self.use_kv_cache = bool(use_kv_cache)

        if positional == "rope":
            self.rope = Rope(dim=self.d_head, max_seq_len=max_seq_len)

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.out_dropout = (
            nn.Dropout(self.out_dropout_p) if self.out_dropout_p > 0.0 else nn.Identity()
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
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
        kv_cache=None,
        layer_idx: int | None = None,
        is_decode: bool = False,
        step_cache=None,
        step_idx: int | None = None,
    ) -> torch.Tensor:
        if context is None:
            context = x

        _ = step_cache
        _ = step_idx

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        if key_padding_mask is not None:
            B = context.shape[0]
            S = context.shape[1]
            check_key_padding_mask(key_padding_mask, batch=B, seqlen=S)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.rope is not None and q.shape[-2] == k.shape[-2]:
            q, k = self.rope.apply(q, k, pos_offset=pos_offset)

        cache_active = self.use_kv_cache and kv_cache is not None
        k_all = k
        v_all = v
        kernel_is_causal = is_causal

        if cache_active:
            if layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided and use_kv_cache=True.")

            if is_decode:
                k_cached, v_cached, cache_len = kv_cache.get(layer_idx)
                if cache_len != pos_offset:
                    raise ValueError(
                        f"KV cache length ({cache_len}) does not match pos_offset ({pos_offset})."
                    )
                if k_cached is not None:
                    if v_cached is None:
                        raise ValueError("Invalid cache state: k exists but v is None.")
                    k_all = torch.cat([k_cached, k], dim=2)
                    v_all = torch.cat([v_cached, v], dim=2)
                kv_cache.append(layer_idx, k, v)
                kernel_is_causal = False
            else:
                kv_cache.layers[layer_idx].k = k
                kv_cache.layers[layer_idx].v = v

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be a boolean tensor (True=keep, False=mask).")
            pad = key_padding_mask.unsqueeze(1).unsqueeze(1)

            if attn_mask is None:
                attn_mask = pad
            else:
                if attn_mask.dtype == torch.bool:
                    attn_mask = attn_mask & pad
                else:
                    mask_value = -1e4 if q.dtype in (torch.float16, torch.bfloat16) else torch.finfo(q.dtype).min
                    additive_pad = torch.zeros_like(pad, dtype=q.dtype).masked_fill(~pad, mask_value)
                    attn_mask = attn_mask + additive_pad

        y = attention_forward(
            q,
            k_all,
            v_all,
            attn_mask=attn_mask,
            is_causal=kernel_is_causal,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            scale=self.scale,
            backend=self.backend,
            normalize=self.normalize,
        )

        y = self._merge_heads(y)
        y = self.out_proj(y)
        y = self.out_dropout(y)
        return y
