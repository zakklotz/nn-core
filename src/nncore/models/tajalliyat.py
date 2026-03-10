from __future__ import annotations

import torch
import torch.nn as nn

from nncore.blocks import TajalliyatBlock
from nncore.layers.norm_factory import make_norm
from nncore.models.config import TajalliyatConfig
from nncore.utils.shapes import check_key_padding_mask


class TajalliyatLM(nn.Module):
    def __init__(self, config: TajalliyatConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = (
            nn.Embedding(config.max_seq_len, config.d_model)
            if config.positional == "absolute"
            else None
        )
        self.blocks = nn.ModuleList([TajalliyatBlock(config) for _ in range(config.num_layers)])
        self.final_norm = make_norm(config.norm, config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seqlen = input_ids.shape
        if seqlen > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seqlen} exceeds max_seq_len {self.config.max_seq_len}."
            )

        x = self.tok_emb(input_ids)
        if self.pos_emb is None:
            return x

        positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(batch, seqlen)
        return x + self.pos_emb(positions)

    def branch_scheduler_status(
        self,
        *,
        device: torch.device | str,
        compiled: bool | None = None,
        dtype: torch.dtype | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
    ) -> dict[str, object]:
        if not self.blocks:
            return {
                "configured": self.config.branch_scheduler,
                "resolved": "sequential",
                "fallback_reason": "no_blocks",
                "active_branches": [],
            }
        return self.blocks[0].branch_scheduler_status(
            device=device,
            compiled=compiled,
            dtype=dtype,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            check_key_padding_mask(
                key_padding_mask,
                batch=input_ids.shape[0],
                seqlen=input_ids.shape[1],
            )

        x = self._embed(input_ids)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        if self.config.return_hidden:
            return x
        return self.lm_head(x)
