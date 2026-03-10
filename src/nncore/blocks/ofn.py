from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING

from nncore.layers import MultiheadAttention, build_mlp
from nncore.layers.norm_factory import make_norm
from nncore.utils.shapes import check_key_padding_mask

if TYPE_CHECKING:
    from nncore.models.config import OFNConfig


def _mask_keep(x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    if key_padding_mask is None:
        return x
    check_key_padding_mask(key_padding_mask, batch=x.shape[0], seqlen=x.shape[1])
    return x * key_padding_mask.unsqueeze(-1).to(dtype=x.dtype)


def _mask_keep_field(x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    if key_padding_mask is None:
        return x
    check_key_padding_mask(key_padding_mask, batch=x.shape[0], seqlen=x.shape[1])
    return x * key_padding_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=x.dtype)


class OFNFieldBuilder(nn.Module):
    def __init__(self, config: OFNConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(
            config.d_model,
            config.field.slots * config.field.d_field,
        )
        horizons = torch.tensor(config.field.ema_timescales, dtype=torch.float32)
        alphas = torch.exp(-1.0 / horizons)
        self.register_buffer("ema_alphas", alphas, persistent=False)

    def _ema_scan(self, updates: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, slots, d_field = updates.shape
        alpha = self.ema_alphas.to(device=updates.device, dtype=updates.dtype).view(1, slots, 1)
        state = torch.zeros(batch_size, slots, d_field, device=updates.device, dtype=updates.dtype)
        outputs: list[torch.Tensor] = []
        for idx in range(seq_len):
            state = (alpha * state) + updates[:, idx]
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        updates = self.proj(x).view(
            batch_size,
            seq_len,
            self.config.field.slots,
            self.config.field.d_field,
        )
        if self.config.field.builder == "cumsum":
            return torch.cumsum(updates, dim=1)
        return self._ema_scan(updates)


class OFNLocalBranch(nn.Module):
    def __init__(self, config: OFNConfig) -> None:
        super().__init__()
        hidden = config.operators.local.d_hidden
        kernel_size = config.operators.local.kernel_size
        self.kernel_size = int(kernel_size)
        self.in_proj = nn.Linear(config.d_model, hidden * 2)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=self.kernel_size, padding=0)
        self.out_proj = nn.Linear(hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = _mask_keep(x, key_padding_mask)
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        y = F.silu(gate) * value
        y = F.pad(y.transpose(1, 2), (self.kernel_size - 1, 0))
        y = self.conv(y).transpose(1, 2)
        y = self.out_proj(y)
        y = self.dropout(y)
        return _mask_keep(y, key_padding_mask)


class OFNAttentionBranch(nn.Module):
    def __init__(self, config: OFNConfig) -> None:
        super().__init__()
        self.mode = config.operators.attention.mode
        self.window_size = int(config.operators.attention.window_size)
        self.attn = MultiheadAttention(
            d_model=config.d_model,
            num_heads=config.n_heads,
            attn_dropout_p=config.dropout,
            out_dropout_p=0.0,
            backend=config.attn_backend,
            positional=config.positional,
            max_seq_len=config.max_seq_len,
            use_kv_cache=False,
        )
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        self._mask_cache: dict[tuple[int, str, int], torch.Tensor] = {}

    def _window_mask(self, seq_len: int, *, device: torch.device) -> torch.Tensor | None:
        if self.mode == "full":
            return None
        device_index = -1 if device.index is None else int(device.index)
        cache_key = (int(seq_len), device.type, device_index)
        mask = self._mask_cache.get(cache_key)
        if mask is None or mask.device != device:
            positions = torch.arange(seq_len, device=device)
            distance = positions[:, None] - positions[None, :]
            mask = (distance >= 0) & (distance < self.window_size)
            self._mask_cache[cache_key] = mask
        return mask

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_mask = self._window_mask(x.shape[1], device=x.device)
        y = self.attn(
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=attn_mask is None,
        )
        y = self.dropout(y)
        return _mask_keep(y, key_padding_mask)


class OFNBarzakhMediator(nn.Module):
    def __init__(self, config: OFNConfig, num_branches: int) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.slots = config.field.slots
        self.d_field = config.field.d_field
        d_model = config.d_model
        d_imaginal = config.mediator.d_imaginal
        gate_hidden = config.mediator.gate_hidden
        self.field_norm = nn.LayerNorm(config.field.d_field)
        self.branch_projs = nn.ModuleList(
            [nn.Linear(d_model, d_imaginal) for _ in range(num_branches)]
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model + config.field.d_field, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_branches),
        )
        self.out_proj = nn.Linear(d_imaginal, d_model)
        self.field_feedback = nn.Linear(d_imaginal, self.slots * self.d_field)

    def forward(
        self,
        x_context: torch.Tensor,
        field_summary: torch.Tensor,
        branch_outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(branch_outputs) != self.num_branches:
            raise ValueError("Unexpected OFN branch count for barzakh mediator.")
        gate_input = torch.cat([x_context, self.field_norm(field_summary)], dim=-1)
        weights = torch.softmax(self.gate(gate_input), dim=-1).unsqueeze(-1)
        projected = torch.stack(
            [proj(branch_out) for proj, branch_out in zip(self.branch_projs, branch_outputs)],
            dim=2,
        )
        fused = (projected * weights).sum(dim=2)
        token_update = self.out_proj(fused)
        field_delta = self.field_feedback(fused).view(
            x_context.shape[0],
            x_context.shape[1],
            self.slots,
            self.d_field,
        )
        return token_update, field_delta


class OFNGatedMediator(nn.Module):
    def __init__(self, config: OFNConfig, num_branches: int) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.slots = config.field.slots
        self.d_field = config.field.d_field
        d_model = config.d_model
        d_imaginal = config.mediator.d_imaginal
        gate_hidden = config.mediator.gate_hidden
        self.branch_projs = nn.ModuleList(
            [nn.Linear(d_model, d_imaginal) for _ in range(num_branches)]
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_branches),
        )
        self.out_proj = nn.Linear(d_imaginal, d_model)
        self.field_feedback = nn.Linear(d_imaginal, self.slots * self.d_field)

    def forward(
        self,
        x_context: torch.Tensor,
        field_summary: torch.Tensor,
        branch_outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = field_summary
        if len(branch_outputs) != self.num_branches:
            raise ValueError("Unexpected OFN branch count for gated mediator.")
        weights = torch.softmax(self.gate(x_context), dim=-1).unsqueeze(-1)
        projected = torch.stack(
            [proj(branch_out) for proj, branch_out in zip(self.branch_projs, branch_outputs)],
            dim=2,
        )
        fused = (projected * weights).sum(dim=2)
        token_update = self.out_proj(fused)
        field_delta = self.field_feedback(fused).view(
            x_context.shape[0],
            x_context.shape[1],
            self.slots,
            self.d_field,
        )
        return token_update, field_delta


class OFNBlock(nn.Module):
    def __init__(self, config: OFNConfig) -> None:
        super().__init__()
        self.config = config
        self.norm1 = make_norm(config.norm, config.d_model, config.norm_eps)
        self.norm2 = make_norm(config.norm, config.d_model, config.norm_eps)
        self.field_mix_norm = nn.LayerNorm(config.field.d_field)
        self.field_out_norm = nn.LayerNorm(config.field.d_field)
        self.field_summary_norm = nn.LayerNorm(config.field.d_field)
        self.field_builder = OFNFieldBuilder(config)
        self.field_to_film = nn.Linear(config.field.d_field, config.d_model * 2)

        self.branches = nn.ModuleDict()
        if config.operators.local.enabled:
            self.branches["local"] = OFNLocalBranch(config)
        if config.operators.attention.enabled:
            self.branches["attention"] = OFNAttentionBranch(config)
        self.branch_names = tuple(self.branches.keys())

        num_branches = len(self.branches)
        if config.mediator.mode == "barzakh":
            self.mediator = OFNBarzakhMediator(config, num_branches)
        else:
            self.mediator = OFNGatedMediator(config, num_branches)

        d_ff = int(config.ffn_mult * config.d_model)
        self.ffn = build_mlp(config.d_model, d_ff)
        self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        if config.field.conditioning != "film":
            raise NotImplementedError("OFN field.conditioning='cross_attend' is reserved but not implemented in v1.")

    def _zero_field(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            x.shape[0],
            x.shape[1],
            self.config.field.slots,
            self.config.field.d_field,
            device=x.device,
            dtype=x.dtype,
        )

    def _field_mix(
        self,
        x_norm: torch.Tensor,
        field_in: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.field.enabled:
            zero_field = self._zero_field(x_norm)
            zero_summary = torch.zeros(
                x_norm.shape[0],
                x_norm.shape[1],
                self.config.field.d_field,
                device=x_norm.device,
                dtype=x_norm.dtype,
            )
            return zero_field, zero_summary

        masked_tokens = _mask_keep(x_norm, key_padding_mask)
        fresh_field = _mask_keep_field(self.field_builder(masked_tokens), key_padding_mask)
        mixed_field = _mask_keep_field(self.field_mix_norm(field_in + fresh_field), key_padding_mask)
        field_summary = self.field_summary_norm(mixed_field.mean(dim=2))
        return mixed_field, field_summary

    def _condition_tokens(self, x_norm: torch.Tensor, field_summary: torch.Tensor) -> torch.Tensor:
        if not self.config.field.enabled:
            return x_norm
        gamma, beta = self.field_to_film(field_summary).chunk(2, dim=-1)
        return x_norm * (1.0 + gamma) + beta

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_state: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm1(x)
        field_in = self._zero_field(x_norm) if field_state is None else field_state
        field_mix, field_summary = self._field_mix(
            x_norm,
            _mask_keep_field(field_in, key_padding_mask),
            key_padding_mask=key_padding_mask,
        )
        x_cond = self._condition_tokens(x_norm, field_summary)
        branch_outputs = [
            self.branches[name](x_cond, key_padding_mask=key_padding_mask)
            for name in self.branch_names
        ]
        token_update, field_delta = self.mediator(x_norm, field_summary, branch_outputs)
        token_update = _mask_keep(token_update, key_padding_mask)
        x = x + self.resid_dropout(token_update)

        if self.config.field.enabled and self.config.field.feedback:
            field_out = self.field_out_norm(
                field_mix + (self.config.field.feedback_scale * field_delta)
            )
        elif self.config.field.enabled:
            field_out = field_mix
        else:
            field_out = self._zero_field(x_norm)
        field_out = _mask_keep_field(field_out, key_padding_mask)

        h = self.norm2(x)
        h = self.ffn(h)
        h = _mask_keep(h, key_padding_mask)
        x = x + self.resid_dropout(h)
        return x, field_out
