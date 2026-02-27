import torch
import torch.nn as nn

from nncore.blocks import TransformerBlock
from nncore.layers import MultiheadAttention, MLP


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
    ):
        super().__init__()

        norm_style = norm_style.lower().strip()
        if norm_style not in ("pre", "post"):
            raise ValueError(f"norm_style must be 'pre' or 'post', got {norm_style!r}")
        self.norm_style = norm_style

        if mlp_dims is None:
            mlp_dims = [d_model, 4 * d_model, d_model]

        # Norms: one per sublayer
        self.ln_sa = nn.LayerNorm(d_model)
        self.ln_ca = nn.LayerNorm(d_model)
        self.ln_ff = nn.LayerNorm(d_model)

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
    ) -> torch.Tensor:
        if self.norm_style == "pre":
            # Self-attention (causal)
            h = self.ln_sa(x)
            h = self.self_attn(
                h,
                attn_mask=self_attn_mask,
                key_padding_mask=self_key_padding_mask,
                is_causal=True,
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
        )
        x = self.ln_sa(x + self.resid_dropout(h))

        h = self.cross_attn(
            x,
            context=enc_out,
            attn_mask=cross_attn_mask,
            key_padding_mask=enc_key_padding_mask,
            is_causal=False,
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

    Teacher-forcing only in nn-core:
      - encoder: forward(src_ids) -> enc_hidden
      - decoder: forward(tgt_ids) -> logits or hidden
      - seq2seq: forward(src_ids, tgt_ids) -> logits

    Generation utilities belong in the harness.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        num_encoder_layers: int = 0,
        num_decoder_layers: int = 0,
        mlp_dims: list[int] | None = None,
        norm_style: str = "pre",
        attn_backend: str = "manual",
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        bias: bool = True,
        attn_scale: float | None = None,
        attn_normalize=None,
        tie_weights: bool = True,
        return_hidden: bool = False,  # if True, return hidden states instead of logits (decoder/seq2seq)
    ):
        super().__init__()
        if num_encoder_layers == 0 and num_decoder_layers == 0:
            raise ValueError("At least one of num_encoder_layers or num_decoder_layers must be > 0.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.return_hidden = bool(return_hidden)

        # Embeddings (shared for simplicity)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Encoder stack (non-causal)
        self.encoder = None
        if self.num_encoder_layers > 0:
            self.encoder = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=d_model,
                        num_heads=num_heads,
                        mlp_dims=mlp_dims,
                        norm_style=norm_style,
                        attn_backend=attn_backend,
                        attn_dropout_p=attn_dropout_p,
                        resid_dropout_p=resid_dropout_p,
                        bias=bias,
                        attn_scale=attn_scale,
                        attn_normalize=attn_normalize,
                    )
                    for _ in range(self.num_encoder_layers)
                ]
            )
            self.enc_final_norm = nn.LayerNorm(d_model)
        else:
            self.enc_final_norm = None

        # Decoder stack (causal self-attn + cross-attn if encoder exists)
        self.decoder = None
        if self.num_decoder_layers > 0:
            if self.num_encoder_layers > 0:
                self.decoder = nn.ModuleList(
                    [
                        TransformerDecoderBlock(
                            d_model=d_model,
                            num_heads=num_heads,
                            mlp_dims=mlp_dims,
                            norm_style=norm_style,
                            attn_backend=attn_backend,
                            attn_dropout_p=attn_dropout_p,
                            resid_dropout_p=resid_dropout_p,
                            bias=bias,
                            attn_scale=attn_scale,
                            attn_normalize=attn_normalize,
                        )
                        for _ in range(self.num_decoder_layers)
                    ]
                )
            else:
                # decoder-only can reuse TransformerBlock with is_causal=True
                self.decoder = nn.ModuleList(
                    [
                        TransformerBlock(
                            d_model=d_model,
                            num_heads=num_heads,
                            mlp_dims=mlp_dims,
                            norm_style=norm_style,
                            attn_backend=attn_backend,
                            attn_dropout_p=attn_dropout_p,
                            resid_dropout_p=resid_dropout_p,
                            bias=bias,
                            attn_scale=attn_scale,
                            attn_normalize=attn_normalize,
                        )
                        for _ in range(self.num_decoder_layers)
                    ]
                )

            self.dec_final_norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

            if tie_weights:
                self.lm_head.weight = self.tok_emb.weight
        else:
            self.dec_final_norm = None
            self.lm_head = None

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, T)
        B, T = ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}.")
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        return self.tok_emb(ids) + self.pos_emb(pos)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor | None = None,
        *,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ):
        """
        Modes:
          - encoder-only: tgt_ids is None and num_decoder_layers == 0
              returns enc_hidden (B, S, d_model)
          - decoder-only: tgt_ids is None and num_encoder_layers == 0
              returns logits (B, T, vocab) or hidden if return_hidden
          - seq2seq: tgt_ids provided and both stacks exist
              returns logits (B, T, vocab) or hidden if return_hidden
        """
        has_encoder = self.encoder is not None
        has_decoder = self.decoder is not None

        if has_encoder and not has_decoder:
            # Encoder-only
            x = self._embed(src_ids)
            for blk in self.encoder:
                x = blk(x, key_padding_mask=src_key_padding_mask, is_causal=False)
            x = self.enc_final_norm(x) if self.enc_final_norm is not None else x
            return x

        if has_decoder and not has_encoder:
            # Decoder-only (GPT-like). src_ids is the decoder input ids here.
            x = self._embed(src_ids)
            for blk in self.decoder:
                x = blk(x, key_padding_mask=tgt_key_padding_mask, is_causal=True)
            x = self.dec_final_norm(x) if self.dec_final_norm is not None else x
            if self.return_hidden:
                return x
            return self.lm_head(x)

        # Seq2seq requires tgt_ids
        if tgt_ids is None:
            raise ValueError("tgt_ids must be provided for seq2seq mode (encoder+decoder).")

        # Encode
        enc = self._embed(src_ids)
        for blk in self.encoder:
            enc = blk(enc, key_padding_mask=src_key_padding_mask, is_causal=False)
        enc = self.enc_final_norm(enc) if self.enc_final_norm is not None else enc

        # Decode with cross-attn
        dec = self._embed(tgt_ids)
        for blk in self.decoder:
            if isinstance(blk, TransformerDecoderBlock):
                dec = blk(
                    dec,
                    enc_out=enc,
                    self_key_padding_mask=tgt_key_padding_mask,
                    enc_key_padding_mask=src_key_padding_mask,
                )
            else:
                # Shouldn't happen in seq2seq mode, but keep safe.
                dec = blk(dec, key_padding_mask=tgt_key_padding_mask, is_causal=True)

        dec = self.dec_final_norm(dec) if self.dec_final_norm is not None else dec
        if self.return_hidden:
            return dec
        return self.lm_head(dec)
