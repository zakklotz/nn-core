# src/nncore/train/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import torch


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

    - Does NOT create the model.
    - Does NOT compute the loss by itself.
    - Owns optimization details: backward, scaler (optional), grad clip, optimizer/scheduler step.
    - Caller provides forward_fn(model, batch) -> loss tensor OR {"loss": loss, ...metrics}.

    Typical usage:
        trainer = Trainer(model, optimizer, scheduler=sched, device=device, amp=True, logger=logger)
        out = trainer.train_step(forward_fn, batch)
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

        # AMP scaler (only meaningful on CUDA)
        self.scaler = scaler
        if self.cfg.amp:
            if self.device.type != "cuda":
                # AMP on CPU is not useful; keep it off silently.
                self.cfg.amp = False
                self.scaler = None
            else:
                self.scaler = self.scaler or torch.amp.GradScaler()

        self.global_step = 0
        self._accum_step = 0  # counts micro-steps since last optimizer step

    def _autocast(self):
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else None,
            enabled=self.cfg.amp and self.device.type == "cuda",
        )

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            try:
                self.logger.info(msg)
            except Exception:
                pass

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

    def train_step(self, forward_fn: ForwardFn, batch: Any) -> Dict[str, Any]:
        """
        Runs one *micro-step* (supports grad accumulation). Optimizer/scheduler step happens
        every grad_accum_steps calls.

        Returns a dict with:
          - loss (float)
          - stepped (bool): whether optimizer stepped this call
          - any additional metrics returned by forward_fn (converted to python scalars if possible)
        """
        self.model.train()

        # Move common batch types to device (best-effort, non-invasive)
        batch = self._to_device(batch)

        with self._autocast():
            out = forward_fn(self.model, batch)
            out = self._normalize_forward_out(out)
            loss_t = out["loss"]

            if loss_t.dim() != 0:
                # ensure scalar
                loss_t = loss_t.mean()

            # scale for accumulation
            loss_scaled = loss_t / float(self.cfg.grad_accum_steps)

        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        self._accum_step += 1
        stepped = False

        if self._accum_step >= self.cfg.grad_accum_steps:
            # Optional grad clipping
            if self.cfg.clip_grad_norm is not None:
                if self.scaler is not None:
                    # unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.cfg.clip_grad_norm),
                )

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Scheduler step (common convention: after optimizer step)
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except TypeError:
                    # some schedulers want a metric; caller can handle that externally
                    self.scheduler.step()

            # Zero grads
            self.optimizer.zero_grad(set_to_none=True)

            self._accum_step = 0
            self.global_step += 1
            stepped = True

        # Prepare return dict
        result: Dict[str, Any] = {"loss": float(loss_t.detach().cpu().item()), "stepped": stepped}
        for k, v in out.items():
            if k == "loss":
                continue
            result[k] = self._to_scalar(v)
        return result

    @torch.no_grad()
    def eval_step(self, forward_fn: ForwardFn, batch: Any) -> Dict[str, Any]:
        """
        Runs a forward-only eval step. Does not update weights.
        """
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
        """
        Minimal trainer state (not checkpointing model weights—caller can do that via nncore.io).
        """
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
        """
        Best-effort move to device:
          - torch.Tensor
          - dict[str, ...]
          - (list/tuple) of items
        """
        if torch.is_tensor(batch):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            t = [self._to_device(v) for v in batch]
            return type(batch)(t)
        return batch
