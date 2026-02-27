from __future__ import annotations

import torch
import torch.nn as nn

from nncore.cache.step_cache import StepCache
from nncore.recurrence.exit_router import ExitRouter
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
        exit_router: ExitRouter | None = None,
        exit_state: dict[str, object] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        steps = n_steps if n_steps is not None else self.n_steps_default
        aux_total: dict[str, torch.Tensor] = {}

        frozen: torch.Tensor | None = None
        if exit_router is not None:
            frozen = torch.zeros(h.shape[0], h.shape[1], dtype=torch.bool, device=h.device)

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

            h_rule = self.rule(h, h_update, step_idx, state=None)

            if exit_router is not None:
                exit_mask_step = exit_router(h, step_idx, state=exit_state)
                if exit_mask_step.dtype != torch.bool:
                    exit_mask_step = exit_mask_step.bool()
                if exit_mask_step.shape != (h.shape[0], h.shape[1]):
                    raise ValueError(
                        f"exit_router must return shape (B,T)=({h.shape[0]},{h.shape[1]}), got {tuple(exit_mask_step.shape)}"
                    )
                frozen = frozen | exit_mask_step
                h = torch.where(frozen.unsqueeze(-1), h, h_rule)
            else:
                h = h_rule

            if return_aux:
                for k, v in aux_step.items():
                    aux_total[k] = aux_total[k] + v if k in aux_total else v

        if return_aux and frozen is not None:
            aux_total["recurrence/exit_frac"] = frozen.float().mean()

        return (h, aux_total) if return_aux else h
