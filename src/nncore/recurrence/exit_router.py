from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class ExitRouter(Protocol):
    def __call__(
        self,
        h: torch.Tensor,
        step_idx: int,
        *,
        state: dict[str, object] | None = None,
    ) -> torch.Tensor:
        """
        Returns exit_mask: BoolTensor[B,T]
        True means 'freeze this token' (stop updating).
        """
        ...


class NullExitRouter(nn.Module):
    def forward(
        self,
        h: torch.Tensor,
        step_idx: int,
        *,
        state: dict[str, object] | None = None,
    ) -> torch.Tensor:
        return torch.zeros(h.shape[0], h.shape[1], dtype=torch.bool, device=h.device)
