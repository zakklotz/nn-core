from __future__ import annotations

import torch

from nncore.constraints.registry import register_constraint


@register_constraint("null")
class NullConstraint:
    def compute(
        self,
        *,
        model: object,
        batch: dict[str, object],
        outputs: dict[str, object] | object,
        step: int | None = None,
        state: dict[str, object] | None = None,
    ) -> dict[str, torch.Tensor]:
        return {}
