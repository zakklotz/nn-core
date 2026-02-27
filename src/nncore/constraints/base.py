from __future__ import annotations

from typing import Protocol

import torch


class Constraint(Protocol):
    def compute(
        self,
        *,
        model: object,
        batch: dict[str, object],
        outputs: dict[str, object] | object,
        step: int | None = None,
        state: dict[str, object] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Return a dict of named scalar losses (each a Tensor scalar).
        Keys should be stable strings like "constraint/foo".
        """
        ...
