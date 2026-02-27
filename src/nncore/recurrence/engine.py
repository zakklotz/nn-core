from __future__ import annotations

import torch
import torch.nn as nn

from nncore.cache.step_cache import StepCache
from nncore.recurrence.rules import UpdateRule


class RecurrenceEngine(nn.Module):
    def __init__(self, block: nn.Module, rule: UpdateRule, n_steps_default: int = 1):
        super().__init__()
        self.block = block
        self.rule = rule
        self.n_steps_default = n_steps_default

    def forward(
        self,
        h: torch.Tensor,
        *,
        n_steps: int | None = None,
        step_cache: StepCache | None = None,
        return_aux: bool = False,
        block_kwargs: dict[str, object] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        steps = n_steps if n_steps is not None else self.n_steps_default
        aux_total: dict[str, torch.Tensor] = {}

        for step_idx in range(steps):
            kwargs = dict(block_kwargs or {})
            kwargs["step_cache"] = step_cache
            kwargs["step_idx"] = step_idx

            if return_aux:
                try:
                    out = self.block(h, return_aux=True, **kwargs)
                    h_update, aux_step = out
                except TypeError:
                    h_update = self.block(h, **kwargs)
                    aux_step = {}
            else:
                h_update = self.block(h, **kwargs)
                aux_step = {}

            h = self.rule(h, h_update, step_idx, state=None)

            if return_aux:
                for k, v in aux_step.items():
                    aux_total[k] = aux_total[k] + v if k in aux_total else v

        return (h, aux_total) if return_aux else h
