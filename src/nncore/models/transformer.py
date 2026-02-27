import torch
import torch.nn as nn

from nncore.blocks import TransformerBlock
from nncore.cache import KVCache
from nncore.layers import MultiheadAttention, MLP
from nncore.layers.norm_factory import make_norm
from nncore.models.config import TransformerConfig
from nncore.recurrence import RecurrenceEngine, ResidualRule


class TransformerDecoderBlock(nn.Module):
    """
    Decoder block with self-attention + cross-attention + MLP.

    norm_style:
      - "pre":  x = x + SA(LN(x)); x = x + CA(LN(x), ctx); x = x + MLP(LN(x))
      - "post": x = LN(x + SA(x)); x = LN(x + CA(x, ctx)); x = LN(x + MLP(x))
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
    ):
        super().__init__()

        norm_style = norm_style.lower().strip()
        if norm_style not in ("pre", "post"):
            raise ValueError(f"norm_style must be 'pre' or 'post', got {norm_style!r}")
        self.norm_style = norm_style

        if mlp_dims is None:
            mlp_dims = [d_model, 4 * d_model, d_model]

        # Norms: one per sublayer
        self.ln_sa = make_norm(norm, d_model, norm_eps)
        self.ln_ca = make_norm(norm, d_model, norm_eps)
        self.ln_ff = make_norm(norm, d_model, norm_eps)

        # Self-attention (causal)
        self.self_attn = MultiheadAttention(
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

        # Cross-attention (non-causal, context from encoder)
        self.cross_attn = MultiheadAttention(
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
            use_kv_cache=False,
        )

        self.mlp = MLP(dimensions=mlp_dims)

        self.resid_dropout = (
            nn.Dropout(resid_dropout_p) if resid_dropout_p > 0.0 else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        enc_out: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_key_padding_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        enc_key_padding_mask: torch.Tensor | None = None,
        pos_offset: int = 0,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        is_decode: bool = False,
        step_cache=None,
        step_idx: int | None = None,
    ) -> torch.Tensor:
        if self.norm_style == "pre":
            # Self-attention (causal)
            h = self.ln_sa(x)
            h = self.self_attn(
                h,
                attn_mask=self_attn_mask,
                key_padding_mask=self_key_padding_mask,
                is_causal=True,
                pos_offset=pos_offset,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                is_decode=is_decode,
                step_cache=step_cache,
                step_idx=step_idx,
            )
            x = x + self.resid_dropout(h)

            # Cross-attention (not causal)
            h = self.ln_ca(x)
            h = self.cross_attn(
                h,
                context=enc_out,
                attn_mask=cross_attn_mask,
                key_padding_mask=enc_key_padding_mask,
                is_causal=False,
                pos_offset=0,
            )
            x = x + self.resid_dropout(h)

            # Feedforward
            h = self.ln_ff(x)
            h = self.mlp(h)
            x = x + self.resid_dropout(h)
            return x

        # Post-norm
        h = self.self_attn(
            x,
            attn_mask=self_attn_mask,
            key_padding_mask=self_key_padding_mask,
            is_causal=True,
            pos_offset=pos_offset,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            is_decode=is_decode,
            step_cache=step_cache,
            step_idx=step_idx,
        )
        x = self.ln_sa(x + self.resid_dropout(h))

        h = self.cross_attn(
            x,
            context=enc_out,
            attn_mask=cross_attn_mask,
            key_padding_mask=enc_key_padding_mask,
            is_causal=False,
            pos_offset=0,
        )
        x = self.ln_ca(x + self.resid_dropout(h))

        h = self.mlp(x)
        x = self.ln_ff(x + self.resid_dropout(h))
        return x


class Transformer(nn.Module):
    """
    One model that can act as:
      - encoder-only
      - decoder-only
      - seq2seq (encoder + decoder)

    Teacher-forcing forward only (generation belongs in harness).

    New API: Transformer(config=TransformerConfig(...), attn_normalize=...)
    Old API still supported via keyword args for backwards compatibility.
    """

    def __init__(
        self,
        config: TransformerConfig | None = None,
        *,
        # --- legacy kwargs (used only if config is None) ---
        vocab_size: int | None = None,
        d_model: int | None = None,
        num_heads: int | None = None,
        max_seq_len: int | None = None,
        num_encoder_layers: int = 0,
        num_decoder_layers: int = 0,
        mlp_dims: list[int] | None = None,
        norm_style: str = "pre",
        attn_backend: str = "manual",
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        bias: bool = True,
        attn_scale: float | None = None,
        tie_weights: bool = True,
        return_hidden: bool = False,
        # --- non-serializable runtime injection ---
        attn_normalize=None,
    ):
        super().__init__()

        if config is None:
            if vocab_size is None or d_model is None or num_heads is None or max_seq_len is None:
                raise ValueError(
                    "When config is None, you must provide vocab_size, d_model, num_heads, max_seq_len."
                )
            config = TransformerConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                tie_weights=tie_weights,
                return_hidden=return_hidden,
            )
            # map legacy into nested configs
            config.attn.backend = attn_backend
            config.attn.attn_backend = attn_backend
            config.attn.dropout_p = float(attn_dropout_p)
            config.attn.resid_dropout_p = float(resid_dropout_p)
            config.attn.scale = attn_scale
            config.block.norm_style = norm_style
            config.block.mlp_dims = mlp_dims
            config.block.bias = bool(bias)

        # config is now the source of truth
        self.config = config

        if self.config.num_encoder_layers == 0 and self.config.num_decoder_layers == 0:
            raise ValueError("At least one of num_encoder_layers or num_decoder_layers must be > 0.")

        # Embeddings (shared for simplicity)
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.pos_emb = nn.Embedding(self.config.max_seq_len, self.config.d_model)

        # Encoder stack (non-causal)
        self.encoder = None
        self.encoder_engine = None
        if self.config.num_encoder_layers > 0:
            if self.config.recursive:
                shared_block = TransformerBlock(
                    d_model=self.config.d_model,
                    num_heads=self.config.num_heads,
                    mlp_dims=self.config.block.mlp_dims,
                    norm_style=self.config.block.norm_style,
                    attn_backend=self.config.attn.attn_backend,
                    attn_dropout_p=self.config.attn.dropout_p,
                    resid_dropout_p=self.config.attn.resid_dropout_p,
                    bias=self.config.block.bias,
                    attn_scale=self.config.attn.scale,
                    attn_normalize=attn_normalize,
                    norm=self.config.block.norm,
                    norm_eps=self.config.block.norm_eps,
                    positional=self.config.positional,
                    max_seq_len=self.config.max_seq_len,
                    use_kv_cache=False,
                    ffn_type=self.config.block.ffn_type,
                    moe_cfg=self.config.block.moe,
                )
                self.encoder_engine = RecurrenceEngine(
                    block=shared_block,
                    rule=ResidualRule(),
                    n_steps_default=self.config.recurrence_steps,
                )
            else:
                self.encoder = nn.ModuleList(
                    [
                        TransformerBlock(
                            d_model=self.config.d_model,
                            num_heads=self.config.num_heads,
                            mlp_dims=self.config.block.mlp_dims,
                            norm_style=self.config.block.norm_style,
                            attn_backend=self.config.attn.attn_backend,
                            attn_dropout_p=self.config.attn.dropout_p,
                            resid_dropout_p=self.config.attn.resid_dropout_p,
                            bias=self.config.block.bias,
                            attn_scale=self.config.attn.scale,
                            attn_normalize=attn_normalize,
                            norm=self.config.block.norm,
                            norm_eps=self.config.block.norm_eps,
                            positional=self.config.positional,
                            max_seq_len=self.config.max_seq_len,
                            use_kv_cache=self.config.attn.use_kv_cache,
                            ffn_type=self.config.block.ffn_type,
                            moe_cfg=self.config.block.moe,
                        )
                        for _ in range(self.config.num_encoder_layers)
                    ]
                )
            self.enc_final_norm = make_norm(
                self.config.block.norm,
                self.config.d_model,
                self.config.block.norm_eps,
            )
        else:
            self.enc_final_norm = None

        # Decoder stack
        self.decoder = None
        self.decoder_engine = None
        if self.config.num_decoder_layers > 0:
            if self.config.num_encoder_layers > 0:
                # seq2seq decoder blocks w/ cross-attn
                self.decoder = nn.ModuleList(
                    [
                        TransformerDecoderBlock(
                            d_model=self.config.d_model,
                            num_heads=self.config.num_heads,
                            mlp_dims=self.config.block.mlp_dims,
                            norm_style=self.config.block.norm_style,
                            attn_backend=self.config.attn.attn_backend,
                            attn_dropout_p=self.config.attn.dropout_p,
                            resid_dropout_p=self.config.attn.resid_dropout_p,
                            bias=self.config.block.bias,
                            attn_scale=self.config.attn.scale,
                            attn_normalize=attn_normalize,
                            norm=self.config.block.norm,
                            norm_eps=self.config.block.norm_eps,
                            positional=self.config.positional,
                            max_seq_len=self.config.max_seq_len,
                            use_kv_cache=self.config.attn.use_kv_cache,
                        )
                        for _ in range(self.config.num_decoder_layers)
                    ]
                )
            else:
                # decoder-only can reuse TransformerBlock with is_causal=True
                if self.config.recursive:
                    shared_block = TransformerBlock(
                        d_model=self.config.d_model,
                        num_heads=self.config.num_heads,
                        mlp_dims=self.config.block.mlp_dims,
                        norm_style=self.config.block.norm_style,
                        attn_backend=self.config.attn.attn_backend,
                        attn_dropout_p=self.config.attn.dropout_p,
                        resid_dropout_p=self.config.attn.resid_dropout_p,
                        bias=self.config.block.bias,
                        attn_scale=self.config.attn.scale,
                        attn_normalize=attn_normalize,
                        norm=self.config.block.norm,
                        norm_eps=self.config.block.norm_eps,
                        positional=self.config.positional,
                        max_seq_len=self.config.max_seq_len,
                        use_kv_cache=False,
                        ffn_type=self.config.block.ffn_type,
                        moe_cfg=self.config.block.moe,
                    )
                    self.decoder_engine = RecurrenceEngine(
                        block=shared_block,
                        rule=ResidualRule(),
                        n_steps_default=self.config.recurrence_steps,
                    )
                else:
                    self.decoder = nn.ModuleList(
                        [
                            TransformerBlock(
                                d_model=self.config.d_model,
                                num_heads=self.config.num_heads,
                                mlp_dims=self.config.block.mlp_dims,
                                norm_style=self.config.block.norm_style,
                                attn_backend=self.config.attn.attn_backend,
                                attn_dropout_p=self.config.attn.dropout_p,
                                resid_dropout_p=self.config.attn.resid_dropout_p,
                                bias=self.config.block.bias,
                                attn_scale=self.config.attn.scale,
                                attn_normalize=attn_normalize,
                                norm=self.config.block.norm,
                                norm_eps=self.config.block.norm_eps,
                                positional=self.config.positional,
                                max_seq_len=self.config.max_seq_len,
                                use_kv_cache=self.config.attn.use_kv_cache,
                                ffn_type=self.config.block.ffn_type,
                                moe_cfg=self.config.block.moe,
                            )
                            for _ in range(self.config.num_decoder_layers)
                        ]
                    )

            self.dec_final_norm = make_norm(
                self.config.block.norm,
                self.config.d_model,
                self.config.block.norm_eps,
            )
            self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

            if self.config.tie_weights:
                self.lm_head.weight = self.tok_emb.weight
        else:
            self.dec_final_norm = None
            self.lm_head = None

    def _embed(self, ids: torch.Tensor, *, pos_offset: int = 0) -> torch.Tensor:
        # ids: (B, T)
        B, T = ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}.")
        x = self.tok_emb(ids)
        if self.config.positional == "absolute":
            pos = torch.arange(pos_offset, pos_offset + T, device=ids.device).unsqueeze(0).expand(B, T)
            return x + self.pos_emb(pos)
        return x

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor | None = None,
        *,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        is_decode: bool = False,
        step_cache=None,
        step_idx: int | None = None,
        return_aux: bool = False,
    ):
        pos_offset = 0
        if kv_cache is not None and is_decode:
            _, _, pos_offset = kv_cache.get(0)

        aux_totals: dict[str, torch.Tensor] = {}

        has_encoder = (self.encoder is not None) or (self.encoder_engine is not None)
        has_decoder = (self.decoder is not None) or (self.decoder_engine is not None)

        if has_encoder and not has_decoder:
            # Encoder-only
            x = self._embed(src_ids, pos_offset=pos_offset)
            if self.encoder_engine is not None:
                out = self.encoder_engine(
                    x,
                    step_cache=step_cache,
                    return_aux=return_aux,
                    block_kwargs={
                        "key_padding_mask": src_key_padding_mask,
                        "is_causal": False,
                        "pos_offset": pos_offset,
                    },
                )
                if return_aux:
                    x, rec_aux = out
                    for k, v in rec_aux.items():
                        aux_totals[k] = aux_totals.get(k, 0.0) + v
                else:
                    x = out
            else:
                for i, blk in enumerate(self.encoder):
                    out = blk(
                        x,
                        key_padding_mask=src_key_padding_mask,
                        is_causal=False,
                        pos_offset=pos_offset,
                        kv_cache=kv_cache,
                        layer_idx=i,
                        is_decode=is_decode,
                        step_cache=step_cache,
                        step_idx=step_idx,
                        return_aux=return_aux,
                    )
                    if return_aux:
                        x, blk_aux = out
                        for k, v in blk_aux.items():
                            aux_totals[k] = aux_totals.get(k, 0.0) + v
                    else:
                        x = out
            x = self.enc_final_norm(x) if self.enc_final_norm is not None else x
            return (x, aux_totals) if return_aux else x

        if has_decoder and not has_encoder:
            # Decoder-only (GPT-like). src_ids is the decoder input ids here.
            x = self._embed(src_ids, pos_offset=pos_offset)
            if self.decoder_engine is not None:
                out = self.decoder_engine(
                    x,
                    step_cache=step_cache,
                    return_aux=return_aux,
                    block_kwargs={
                        "key_padding_mask": tgt_key_padding_mask,
                        "is_causal": True,
                        "pos_offset": pos_offset,
                    },
                )
                if return_aux:
                    x, rec_aux = out
                    for k, v in rec_aux.items():
                        aux_totals[k] = aux_totals.get(k, 0.0) + v
                else:
                    x = out
            else:
                for i, blk in enumerate(self.decoder):
                    out = blk(
                        x,
                        key_padding_mask=tgt_key_padding_mask,
                        is_causal=True,
                        pos_offset=pos_offset,
                        kv_cache=kv_cache,
                        layer_idx=i,
                        is_decode=is_decode,
                        step_cache=step_cache,
                        step_idx=step_idx,
                        return_aux=return_aux,
                    )
                    if return_aux:
                        x, blk_aux = out
                        for k, v in blk_aux.items():
                            aux_totals[k] = aux_totals.get(k, 0.0) + v
                    else:
                        x = out
            x = self.dec_final_norm(x) if self.dec_final_norm is not None else x
            if self.config.return_hidden:
                return (x, aux_totals) if return_aux else x
            logits = self.lm_head(x)
            return (logits, aux_totals) if return_aux else logits

        # Seq2seq requires tgt_ids
        if tgt_ids is None:
            raise ValueError("tgt_ids must be provided for seq2seq mode (encoder+decoder).")

        # Encode
        enc = self._embed(src_ids, pos_offset=0)
        if self.encoder_engine is not None:
            out = self.encoder_engine(
                enc,
                step_cache=step_cache,
                return_aux=return_aux,
                block_kwargs={
                    "key_padding_mask": src_key_padding_mask,
                    "is_causal": False,
                    "pos_offset": 0,
                },
            )
            if return_aux:
                enc, rec_aux = out
                for k, v in rec_aux.items():
                    aux_totals[k] = aux_totals.get(k, 0.0) + v
            else:
                enc = out
        else:
            for i, blk in enumerate(self.encoder):
                out = blk(
                    enc,
                    key_padding_mask=src_key_padding_mask,
                    is_causal=False,
                    pos_offset=0,
                    return_aux=return_aux,
                )
                if return_aux:
                    enc, blk_aux = out
                    for k, v in blk_aux.items():
                        aux_totals[k] = aux_totals.get(k, 0.0) + v
                else:
                    enc = out
        enc = self.enc_final_norm(enc) if self.enc_final_norm is not None else enc

        # Decode with cross-attn
        dec = self._embed(tgt_ids, pos_offset=pos_offset)
        for i, blk in enumerate(self.decoder):
            if isinstance(blk, TransformerDecoderBlock):
                dec = blk(
                    dec,
                    enc_out=enc,
                    self_key_padding_mask=tgt_key_padding_mask,
                    enc_key_padding_mask=src_key_padding_mask,
                    pos_offset=pos_offset,
                    kv_cache=kv_cache,
                    layer_idx=i,
                    is_decode=is_decode,
                    step_cache=step_cache,
                    step_idx=step_idx,
                )
            else:
                out = blk(
                    dec,
                    key_padding_mask=tgt_key_padding_mask,
                    is_causal=True,
                    pos_offset=pos_offset,
                    kv_cache=kv_cache,
                    layer_idx=i,
                    is_decode=is_decode,
                    step_cache=step_cache,
                    step_idx=step_idx,
                    return_aux=return_aux,
                )
                if return_aux:
                    dec, blk_aux = out
                    for k, v in blk_aux.items():
                        aux_totals[k] = aux_totals.get(k, 0.0) + v
                else:
                    dec = out

        dec = self.dec_final_norm(dec) if self.dec_final_norm is not None else dec
        if self.config.return_hidden:
            return (dec, aux_totals) if return_aux else dec
        logits = self.lm_head(dec)
        return (logits, aux_totals) if return_aux else logits
