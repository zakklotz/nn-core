from __future__ import annotations

from typing import Protocol

import torch


class Hook(Protocol):
    def on_hidden(
        self,
        h: torch.Tensor,
        *,
        step: int | None = None,
        state: dict[str, object] | None = None,
    ) -> torch.Tensor:
        return h

    def on_logits(
        self,
        logits: torch.Tensor,
        *,
        step: int | None = None,
        state: dict[str, object] | None = None,
    ) -> torch.Tensor:
        return logits

    def on_loss(
        self,
        loss_dict: dict[str, torch.Tensor],
        *,
        step: int | None = None,
        state: dict[str, object] | None = None,
    ) -> dict[str, torch.Tensor]:
        return loss_dict
