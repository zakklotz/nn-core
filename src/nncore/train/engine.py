# src/nncore/train/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import torch

from nncore.constraints import build_constraint

ForwardOut = Union[torch.Tensor, Dict[str, Any]]
ForwardFn = Callable[[torch.nn.Module, Any], ForwardOut]


@dataclass
class TrainerConfig:
    amp: bool = False
    grad_accum_steps: int = 1
    clip_grad_norm: float | None = None


class Trainer:
    """
    Minimal training engine.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device | str = "cpu",
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        logger: Optional[Any] = None,
        amp: bool = False,
        grad_accum_steps: int = 1,
        clip_grad_norm: float | None = None,
    ):
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

        self.device = torch.device(device)
        self.model.to(self.device)

        self.cfg = TrainerConfig(
            amp=bool(amp),
            grad_accum_steps=int(grad_accum_steps),
            clip_grad_norm=clip_grad_norm,
        )

        self.scaler = scaler
        if self.cfg.amp:
            if self.device.type != "cuda":
                self.cfg.amp = False
                self.scaler = None
            else:
                self.scaler = self.scaler or torch.amp.GradScaler()

        self.global_step = 0
        self._accum_step = 0
        self._constraints: list[tuple[object, float]] = []
        self._init_constraints()

    def _init_constraints(self) -> None:
        model_cfg = getattr(self.model, "config", None)
        cfg_constraints = getattr(model_cfg, "constraints", None)
        if not cfg_constraints:
            return
        for entry in cfg_constraints:
            constraint = build_constraint(entry.name, dict(entry.params))
            self._constraints.append((constraint, float(entry.weight)))

    def _autocast(self):
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else None,
            enabled=self.cfg.amp and self.device.type == "cuda",
        )

    def _normalize_forward_out(self, out: ForwardOut) -> Dict[str, Any]:
        if isinstance(out, torch.Tensor):
            return {"loss": out}
        elif isinstance(out, dict):
            if "loss" not in out:
                raise ValueError("forward_fn returned a dict but missing required key 'loss'.")
            if not isinstance(out["loss"], torch.Tensor):
                raise ValueError("forward_fn dict['loss'] must be a torch.Tensor.")
            return out
        raise TypeError("forward_fn must return a loss tensor or a dict containing a 'loss' tensor.")

    def _compute_constraint_losses(
        self,
        *,
        batch: Any,
        outputs: Dict[str, Any],
        step: int,
    ) -> dict[str, torch.Tensor]:
        totals: dict[str, torch.Tensor] = {}
        if not self._constraints:
            return totals

        for constraint, weight in self._constraints:
            losses = constraint.compute(
                model=self.model,
                batch=batch,
                outputs=outputs,
                step=step,
                state=None,
            )
            if not isinstance(losses, dict):
                raise ValueError("Constraint.compute must return dict[str, Tensor].")
            for key, value in losses.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"Constraint loss {key!r} must be a torch.Tensor.")
                if value.dim() != 0:
                    raise ValueError(f"Constraint loss {key!r} must be a scalar tensor.")
                if value.device != self.device:
                    raise ValueError(
                        f"Constraint loss {key!r} is on {value.device}, expected {self.device}."
                    )
                weighted = value * float(weight)
                totals[key] = totals[key] + weighted if key in totals else weighted

        return totals

    def train_step(self, forward_fn: ForwardFn, batch: Any) -> Dict[str, Any]:
        self.model.train()
        batch = self._to_device(batch)

        with self._autocast():
            out = forward_fn(self.model, batch)
            out = self._normalize_forward_out(out)
            loss_t = out["loss"]

            if loss_t.dim() != 0:
                loss_t = loss_t.mean()

            constraint_losses = self._compute_constraint_losses(
                batch=batch,
                outputs=out,
                step=self.global_step,
            )
            if constraint_losses:
                loss_t = loss_t + sum(constraint_losses.values())
                out.update(constraint_losses)

            loss_scaled = loss_t / float(self.cfg.grad_accum_steps)

        if self.scaler is not None:
            self.scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        self._accum_step += 1
        stepped = False

        if self._accum_step >= self.cfg.grad_accum_steps:
            if self.cfg.clip_grad_norm is not None:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.cfg.clip_grad_norm),
                )

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except TypeError:
                    self.scheduler.step()

            self.optimizer.zero_grad(set_to_none=True)

            self._accum_step = 0
            self.global_step += 1
            stepped = True

        result: Dict[str, Any] = {"loss": float(loss_t.detach().cpu().item()), "stepped": stepped}
        for k, v in out.items():
            if k == "loss":
                continue
            result[k] = self._to_scalar(v)
        return result

    @torch.no_grad()
    def eval_step(self, forward_fn: ForwardFn, batch: Any) -> Dict[str, Any]:
        self.model.eval()
        batch = self._to_device(batch)

        with self._autocast():
            out = forward_fn(self.model, batch)
            out = self._normalize_forward_out(out)
            loss_t = out["loss"]
            if loss_t.dim() != 0:
                loss_t = loss_t.mean()

        result: Dict[str, Any] = {"loss": float(loss_t.detach().cpu().item())}
        for k, v in out.items():
            if k == "loss":
                continue
            result[k] = self._to_scalar(v)
        return result

    def state_dict(self) -> Dict[str, Any]:
        sd: Dict[str, Any] = {
            "global_step": self.global_step,
            "accum_step": self._accum_step,
        }
        if self.scaler is not None:
            sd["scaler"] = self.scaler.state_dict()
        return sd

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.global_step = int(state.get("global_step", 0))
        self._accum_step = int(state.get("accum_step", 0))
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

    def _to_scalar(self, v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return float(v.detach().cpu().item())
            return v.detach().cpu()
        return v

    def _to_device(self, batch: Any) -> Any:
        if torch.is_tensor(batch):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            t = [self._to_device(v) for v in batch]
            return type(batch)(t)
        return batch
