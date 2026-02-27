# src/nncore/blocks/transformer.py

import torch
import torch.nn as nn

from nncore.layers import MultiheadAttention, MLP


class TransformerBlock(nn.Module):
    """
    Transformer block with configurable LayerNorm style.

    norm_style:
      - "pre":  x = x + Attn(LN(x)); x = x + MLP(LN(x))   (default, more stable)
      - "post": x = LN(x + Attn(x)); x = LN(x + MLP(x))   (classic)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        mlp_dims: list[int] | None = None,
        norm_style: str = "pre",          # "pre" or "post"
        attn_backend: str = "manual",     # "manual" or "sdpa"
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        bias: bool = True,
        attn_scale: float | None = None,
        attn_normalize=None,             # callable(scores)->weights, only for manual backend
    ):
        super().__init__()

        norm_style = norm_style.lower().strip()
        if norm_style not in ("pre", "post"):
            raise ValueError(f"norm_style must be 'pre' or 'post', got {norm_style!r}")
        self.norm_style = norm_style

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attn = MultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            bias=bias,
            attn_dropout_p=attn_dropout_p,
            out_dropout_p=resid_dropout_p,
            backend=attn_backend,
            scale=attn_scale,
            normalize=attn_normalize,
        )

        # Default transformer FFN dims: [d_model, 4*d_model, d_model]
        if mlp_dims is None:
            mlp_dims = [d_model, 4 * d_model, d_model]
        self.mlp = MLP(dimensions=mlp_dims)

        self.resid_dropout = (
            nn.Dropout(resid_dropout_p) if resid_dropout_p > 0.0 else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        context: torch.Tensor | None = None,   # for cross-attn if desired
    ) -> torch.Tensor:
        if self.norm_style == "pre":
            # Attention block (pre-norm)
            h = self.ln1(x)
            h = self.attn(
                h,
                context=context,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self.resid_dropout(h)

            # MLP block (pre-norm)
            h = self.ln2(x)
            h = self.mlp(h)
            x = x + self.resid_dropout(h)
            return x

        # Post-norm
        h = self.attn(
            x,
            context=context,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        x = self.ln1(x + self.resid_dropout(h))

        h = self.mlp(x)
        x = self.ln2(x + self.resid_dropout(h))
        return x
