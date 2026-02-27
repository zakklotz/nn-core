from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F


@dataclass
class ToyLMConfig:
    vocab_size: int = 128
    seq_len: int = 64
    batch_size: int = 16


def make_toy_lm_batch(
    cfg: ToyLMConfig,
    *,
    device: torch.device | str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Produces a synthetic next-token-prediction batch.

    Returns dict:
      - input_ids: (B, T) int64 tokens
    Labels are implied as input_ids shifted by 1 in the forward_fn.
    """
    device = torch.device(device)
    input_ids = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(cfg.batch_size, cfg.seq_len),
        device=device,
        dtype=torch.long,
    )
    return {"input_ids": input_ids}


def toy_lm_forward_fn(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Forward function for Trainer.

    Expects a decoder-only Transformer-like model that returns logits:
      logits: (B, T, V)

    Computes next-token cross-entropy:
      loss = CE(logits[:, :-1, :], input_ids[:, 1:])
    """
    input_ids = batch["input_ids"]
    logits = model(input_ids)  # decoder-only mode in nncore.models.Transformer

    # Next-token prediction: predict token t+1 from position t
    logits = logits[:, :-1, :]              # (B, T-1, V)
    targets = input_ids[:, 1:]              # (B, T-1)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

    return {
        "loss": loss,
        "token_ce": loss.detach(),
    }
