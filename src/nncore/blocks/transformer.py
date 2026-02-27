# src/nncore/blocks/transformer.py

import torch
import torch.nn as nn

from nncore.layers import MultiheadAttention, MLP
from nncore.layers.norm_factory import make_norm
from nncore.models.config import MoEConfig
from nncore.moe import MoELayer


class TransformerBlock(nn.Module):
    """
    Transformer block with configurable LayerNorm style.

    norm_style:
      - "pre":  x = x + Attn(LN(x)); x = x + FFN(LN(x))
      - "post": x = LN(x + Attn(x)); x = LN(x + FFN(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        mlp_dims: list[int] | None = None,
        norm_style: str = "pre",
        attn_backend: str = "manual",
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        bias: bool = True,
        attn_scale: float | None = None,
        attn_normalize=None,
        norm: str = "layernorm",
        norm_eps: float = 1e-5,
        positional: str = "absolute",
        max_seq_len: int = 2048,
        use_kv_cache: bool = False,
        ffn_type: str = "mlp",
        moe_cfg: MoEConfig | None = None,
    ):
        super().__init__()

        norm_style = norm_style.lower().strip()
        if norm_style not in ("pre", "post"):
            raise ValueError(f"norm_style must be 'pre' or 'post', got {norm_style!r}")
        self.norm_style = norm_style

        self.ln1 = make_norm(norm, d_model, norm_eps)
        self.ln2 = make_norm(norm, d_model, norm_eps)

        self.attn = MultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            bias=bias,
            attn_dropout_p=attn_dropout_p,
            out_dropout_p=resid_dropout_p,
            backend=attn_backend,
            scale=attn_scale,
            normalize=attn_normalize,
            positional=positional,
            max_seq_len=max_seq_len,
            use_kv_cache=use_kv_cache,
        )

        if mlp_dims is None:
            mlp_dims = [d_model, 4 * d_model, d_model]

        self.ffn_type = ffn_type
        if self.ffn_type == "mlp":
            self.ffn = MLP(dimensions=mlp_dims)
        elif self.ffn_type == "moe":
            if moe_cfg is None:
                raise ValueError("moe_cfg must be provided when ffn_type='moe'.")
            self.ffn = MoELayer(d_model=d_model, cfg=moe_cfg, mlp_hidden_fallback=mlp_dims[1])
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type!r}")

        self.resid_dropout = (
            nn.Dropout(resid_dropout_p) if resid_dropout_p > 0.0 else nn.Identity()
        )

    def _ffn_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.ffn_type == "moe":
            y, aux = self.ffn(x)
            return y, aux
        return self.ffn(x), {}

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        context: torch.Tensor | None = None,
        pos_offset: int = 0,
        kv_cache=None,
        layer_idx: int | None = None,
        is_decode: bool = False,
        step_cache=None,
        step_idx: int | None = None,
        return_aux: bool = False,
    ):
        aux_losses: dict[str, torch.Tensor] = {}

        if self.norm_style == "pre":
            h = self.ln1(x)
            h = self.attn(
                h,
                context=context,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                pos_offset=pos_offset,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                is_decode=is_decode,
                step_cache=step_cache,
                step_idx=step_idx,
            )
            x = x + self.resid_dropout(h)

            h = self.ln2(x)
            h, ffn_aux = self._ffn_forward(h)
            x = x + self.resid_dropout(h)
            aux_losses.update(ffn_aux)

            return (x, aux_losses) if return_aux else x

        h = self.attn(
            x,
            context=context,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            pos_offset=pos_offset,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            is_decode=is_decode,
            step_cache=step_cache,
            step_idx=step_idx,
        )
        x = self.ln1(x + self.resid_dropout(h))

        h, ffn_aux = self._ffn_forward(x)
        x = self.ln2(x + self.resid_dropout(h))
        aux_losses.update(ffn_aux)

        return (x, aux_losses) if return_aux else x
