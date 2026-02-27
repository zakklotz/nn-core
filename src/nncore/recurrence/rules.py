from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class UpdateRule(Protocol):
    def __call__(
        self,
        h_prev: torch.Tensor,
        h_update: torch.Tensor,
        step_idx: int,
        *,
        state: dict[str, object] | None = None,
    ) -> torch.Tensor:
        ...


@dataclass
class ResidualRule:
    """Standard residual update: h_next = h_prev + h_update"""

    def __call__(
        self,
        h_prev: torch.Tensor,
        h_update: torch.Tensor,
        step_idx: int,
        *,
        state: dict[str, object] | None = None,
    ) -> torch.Tensor:
        return h_prev + h_update
