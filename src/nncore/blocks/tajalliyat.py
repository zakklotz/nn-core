from __future__ import annotations

import time
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, ClassVar

from nncore.layers import MultiheadAttention, build_mlp
from nncore.layers.norm_factory import make_norm
from nncore.utils.shapes import check_key_padding_mask

if TYPE_CHECKING:
    from nncore.models.config import TajalliyatConfig


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling") and compiler.is_compiling():
        return True
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling") and dynamo.is_compiling():
        return True
    return False


def _mask_keep(x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
    if key_padding_mask is None:
        return x
    check_key_padding_mask(key_padding_mask, batch=x.shape[0], seqlen=x.shape[1])
    return x * key_padding_mask.unsqueeze(-1).to(dtype=x.dtype)


class AttentionBranch(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        attn_backend: str = "sdpa",
        dropout: float = 0.0,
        positional: str = "absolute",
        max_seq_len: int = 2048,
        use_output_proj: bool = False,
        branch_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = MultiheadAttention(
            d_model=d_model,
            num_heads=n_heads,
            attn_dropout_p=dropout,
            out_dropout_p=0.0,
            backend=attn_backend,
            positional=positional,
            max_seq_len=max_seq_len,
            use_kv_cache=False,
        )
        self.output_proj = nn.Linear(d_model, d_model) if use_output_proj else nn.Identity()
        self.dropout = nn.Dropout(branch_dropout) if branch_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self.attn(x, key_padding_mask=key_padding_mask, is_causal=True)
        y = self.output_proj(y)
        y = self.dropout(y)
        return _mask_keep(y, key_padding_mask)


class CNNBranch(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        kernel_size: int,
        use_output_proj: bool = False,
        branch_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=self.kernel_size, padding=0)
        self.output_proj = nn.Linear(d_model, d_model) if use_output_proj else nn.Identity()
        self.dropout = nn.Dropout(branch_dropout) if branch_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = _mask_keep(x, key_padding_mask)
        y = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        y = self.conv(y).transpose(1, 2)
        y = self.output_proj(y)
        y = self.dropout(y)
        return _mask_keep(y, key_padding_mask)


class MambaBranch(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        use_output_proj: bool = False,
        branch_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:
            raise ImportError(
                "mamba-ssm is required when use_mamba=True; install mamba-ssm and causal-conv1d"
            ) from exc

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.output_proj = nn.Linear(d_model, d_model) if use_output_proj else nn.Identity()
        self.dropout = nn.Dropout(branch_dropout) if branch_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = _mask_keep(x, key_padding_mask)
        y = self.mamba(x)
        y = self.output_proj(y)
        y = self.dropout(y)
        return _mask_keep(y, key_padding_mask)


class SimpleFusion(nn.Module):
    def forward(self, x: torch.Tensor, branch_outputs: list[torch.Tensor]) -> torch.Tensor:
        _ = x
        if not branch_outputs:
            raise ValueError("SimpleFusion requires at least one branch output.")
        return torch.stack(branch_outputs, dim=0).sum(dim=0)


class GatedFusion(nn.Module):
    def __init__(self, d_model: int, num_branches: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_branches),
        )

    def forward(self, x: torch.Tensor, branch_outputs: list[torch.Tensor]) -> torch.Tensor:
        if not branch_outputs:
            raise ValueError("GatedFusion requires at least one branch output.")
        stacked = torch.stack(branch_outputs, dim=2)
        weights = torch.softmax(self.gate(x), dim=-1).unsqueeze(-1)
        return (stacked * weights).sum(dim=2)


class BarzakhFusion(nn.Module):
    """
    Inputs:
      - x: [B, T, D]
      - branch_outputs[i]: [B, T, D] for i in [1, N]

    Fusion:
      - C = concat(x, b1, ..., bN) with shape [B, T, (N+1)D]
      - m = mediator_mlp(C) with shape [B, T, D]
      - a = softmax(branch_logits(m), dim=-1) with shape [B, T, N]
      - u_i = branch_proj_i(b_i) with shape [B, T, D]
      - fused = mediator_out(m) + sum_i a_i * u_i
    """

    def __init__(self, d_model: int, num_branches: int) -> None:
        super().__init__()
        in_dim = (num_branches + 1) * d_model
        self.mediator_mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.branch_logits = nn.Linear(d_model, num_branches)
        self.mediator_out = nn.Linear(d_model, d_model)
        self.branch_projs = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_branches)]
        )

    def forward(self, x: torch.Tensor, branch_outputs: list[torch.Tensor]) -> torch.Tensor:
        if not branch_outputs:
            raise ValueError("BarzakhFusion requires at least one branch output.")
        concat = torch.cat([x, *branch_outputs], dim=-1)
        mediator = self.mediator_mlp(concat)
        weights = torch.softmax(self.branch_logits(mediator), dim=-1).unsqueeze(-1)
        projected = torch.stack(
            [proj(branch_out) for proj, branch_out in zip(self.branch_projs, branch_outputs)],
            dim=2,
        )
        return self.mediator_out(mediator) + (projected * weights).sum(dim=2)


class TajalliyatBlock(nn.Module):
    _auto_scheduler_cache: ClassVar[dict[tuple[object, ...], tuple[str, str | None]]] = {}

    def __init__(self, config: TajalliyatConfig) -> None:
        super().__init__()
        self.config = config
        self.norm1 = make_norm(config.norm, config.d_model, config.norm_eps)
        self.norm2 = make_norm(config.norm, config.d_model, config.norm_eps)

        self.branches = nn.ModuleDict()
        if config.use_attention:
            self.branches["attention"] = AttentionBranch(
                config.d_model,
                config.n_heads,
                attn_backend=config.attn_backend,
                dropout=config.dropout,
                positional=config.positional,
                max_seq_len=config.max_seq_len,
                use_output_proj=config.attention_branch_proj,
                branch_dropout=config.branch_dropout,
            )
        if config.use_cnn:
            self.branches["cnn"] = CNNBranch(
                config.d_model,
                kernel_size=config.cnn_kernel_size,
                use_output_proj=config.cnn_branch_proj,
                branch_dropout=config.branch_dropout,
            )
        if config.use_mamba:
            self.branches["mamba"] = MambaBranch(
                config.d_model,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                use_output_proj=config.mamba_branch_proj,
                branch_dropout=config.branch_dropout,
            )

        self.branch_names = tuple(self.branches.keys())
        self._stream_cache: dict[tuple[str, int], list[torch.cuda.Stream]] = {}

        num_branches = len(self.branches)
        if config.fusion_type == "sum":
            self.fusion = SimpleFusion()
        elif config.fusion_type == "gated_sum":
            self.fusion = GatedFusion(config.d_model, num_branches)
        else:
            self.fusion = BarzakhFusion(config.d_model, num_branches)

        d_ff = int(config.ffn_mult * config.d_model)
        self.ffn = build_mlp(config.d_model, d_ff)
        self.resid_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def _module_dtype(self) -> torch.dtype:
        param = next(self.parameters(), None)
        return torch.float32 if param is None else param.dtype

    def _module_device(self) -> torch.device:
        param = next(self.parameters(), None)
        return torch.device("cpu") if param is None else param.device

    def _devices_match(self, lhs: torch.device, rhs: torch.device) -> bool:
        if lhs.type != rhs.type:
            return False
        if lhs.type != "cuda":
            return True
        lhs_index = lhs.index if lhs.index is not None else torch.cuda.current_device()
        rhs_index = rhs.index if rhs.index is not None else torch.cuda.current_device()
        return int(lhs_index) == int(rhs_index)

    def _cuda_streams_unavailable_reason(
        self,
        *,
        device: torch.device,
        compiled: bool,
    ) -> str | None:
        if len(self.branch_names) <= 1:
            return "single_active_branch"

        if device.type != "cuda" or not torch.cuda.is_available():
            return "cuda_unavailable"

        if compiled:
            return "compiled_model"

        return None

    def _cuda_streams_error(self, reason: str) -> RuntimeError:
        if reason == "single_active_branch":
            return RuntimeError(
                "Tajalliyat branch_scheduler='cuda_streams' requires at least two active branches."
            )
        if reason == "cuda_unavailable":
            return RuntimeError(
                "Tajalliyat branch_scheduler='cuda_streams' requires CUDA."
            )
        if reason == "compiled_model":
            return RuntimeError(
                "Tajalliyat branch_scheduler='cuda_streams' is not supported under torch.compile."
            )
        return RuntimeError(f"Tajalliyat branch_scheduler='cuda_streams' is unavailable: {reason}")

    def _autocast_context(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    def _auto_scheduler_cache_key(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        seqlen: int,
    ) -> tuple[object, ...]:
        index = device.index if device.index is not None else 0
        return (
            "tajalliyat_auto_scheduler_v1",
            device.type,
            int(index),
            str(dtype),
            int(batch_size),
            int(seqlen),
            bool(self.training),
            self.branch_names,
            int(self.config.d_model),
            int(self.config.n_heads),
            str(self.config.attn_backend),
            str(self.config.positional),
            int(self.config.cnn_kernel_size),
            int(self.config.mamba_d_state),
            int(self.config.mamba_d_conv),
            int(self.config.mamba_expand),
        )

    def _benchmark_branch_scheduler(
        self,
        scheduler: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        seqlen: int,
    ) -> float:
        warmup_iters = 1
        timed_iters = 2
        elapsed_times: list[float] = []
        runner = (
            self._run_branches_cuda_streams
            if scheduler == "cuda_streams"
            else self._run_branches_sequential
        )

        for iter_idx in range(warmup_iters + timed_iters):
            x = torch.randn(
                batch_size,
                seqlen,
                self.config.d_model,
                device=device,
                dtype=torch.float32,
            )
            key_padding_mask = torch.ones(batch_size, seqlen, dtype=torch.bool, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.no_grad():
                with self._autocast_context(device=device, dtype=dtype):
                    outputs = runner(x, key_padding_mask=key_padding_mask)
                    _ = torch.stack([out.float().square().mean() for out in outputs]).sum()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if iter_idx >= warmup_iters:
                elapsed_times.append(max(time.perf_counter() - start, 1e-12))

        return sum(elapsed_times) / len(elapsed_times)

    def _select_auto_scheduler(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        seqlen: int,
    ) -> tuple[str, str | None]:
        if not self._devices_match(self._module_device(), device):
            return "sequential", "auto_benchmark_pending_device_move"

        cache_key = self._auto_scheduler_cache_key(
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            seqlen=seqlen,
        )
        cached = self._auto_scheduler_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            sequential_time = self._benchmark_branch_scheduler(
                "sequential",
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                seqlen=seqlen,
            )
            cuda_streams_time = self._benchmark_branch_scheduler(
                "cuda_streams",
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                seqlen=seqlen,
            )
        except Exception as exc:
            resolved = ("sequential", f"auto_benchmark_failed:{type(exc).__name__}")
            self._auto_scheduler_cache[cache_key] = resolved
            return resolved

        if cuda_streams_time < sequential_time:
            resolved = ("cuda_streams", None)
        else:
            resolved = ("sequential", "auto_benchmark_selected_sequential")
        self._auto_scheduler_cache[cache_key] = resolved
        return resolved

    def _resolve_branch_scheduler(
        self,
        *,
        device: torch.device,
        compiled: bool,
        dtype: torch.dtype | None = None,
        batch_size: int | None = None,
        seqlen: int | None = None,
    ) -> tuple[str, str | None]:
        configured = self.config.branch_scheduler

        if configured == "sequential":
            return "sequential", None

        unavailable_reason = self._cuda_streams_unavailable_reason(
            device=device,
            compiled=compiled,
        )
        if configured == "cuda_streams":
            if unavailable_reason is not None:
                raise self._cuda_streams_error(unavailable_reason)
            return "cuda_streams", None

        if unavailable_reason is not None:
            return "sequential", unavailable_reason

        auto_dtype = self._module_dtype() if dtype is None else dtype
        auto_batch_size = 1 if batch_size is None else max(1, int(batch_size))
        auto_seqlen = (
            min(self.config.max_seq_len, 512)
            if seqlen is None
            else max(1, min(int(seqlen), self.config.max_seq_len))
        )
        return self._select_auto_scheduler(
            device=device,
            dtype=auto_dtype,
            batch_size=auto_batch_size,
            seqlen=auto_seqlen,
        )

    def branch_scheduler_status(
        self,
        *,
        device: torch.device | str,
        compiled: bool | None = None,
        dtype: torch.dtype | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
    ) -> dict[str, object]:
        device_obj = torch.device(device)
        resolved, fallback_reason = self._resolve_branch_scheduler(
            device=device_obj,
            compiled=_is_torch_compiling() if compiled is None else bool(compiled),
            dtype=dtype,
            batch_size=batch_size,
            seqlen=seq_len,
        )
        return {
            "configured": self.config.branch_scheduler,
            "resolved": resolved,
            "fallback_reason": fallback_reason,
            "active_branches": list(self.branch_names),
        }

    def _main_branch_name(self) -> str:
        if "mamba" in self.branches:
            return "mamba"
        if "attention" in self.branches:
            return "attention"
        return self.branch_names[0]

    def _stream_cache_key(self, device: torch.device) -> tuple[str, int]:
        if device.type != "cuda":
            raise RuntimeError("CUDA stream cache requested for non-CUDA device.")
        index = device.index if device.index is not None else torch.cuda.current_device()
        return device.type, int(index)

    def _get_aux_streams(self, device: torch.device, count: int) -> list[torch.cuda.Stream]:
        if count <= 0:
            return []
        cache_key = self._stream_cache_key(device)
        streams = self._stream_cache.setdefault(cache_key, [])
        while len(streams) < count:
            streams.append(torch.cuda.Stream(device=device))
        return streams[:count]

    def _run_branches_sequential(
        self,
        h: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [self.branches[name](h, key_padding_mask=key_padding_mask) for name in self.branch_names]

    def _run_branches_cuda_streams(
        self,
        h: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        device = h.device
        main_stream = torch.cuda.current_stream(device=device)
        main_name = self._main_branch_name()
        side_names = [name for name in self.branch_names if name != main_name]
        side_streams = self._get_aux_streams(device, len(side_names))
        outputs: dict[str, torch.Tensor] = {}

        launched: list[tuple[torch.cuda.Stream, torch.Tensor]] = []
        for name, stream in zip(side_names, side_streams):
            stream.wait_stream(main_stream)
            h.record_stream(stream)
            if key_padding_mask is not None and key_padding_mask.device.type == "cuda":
                key_padding_mask.record_stream(stream)
            with torch.cuda.stream(stream):
                out = self.branches[name](h, key_padding_mask=key_padding_mask)
            outputs[name] = out
            launched.append((stream, out))

        outputs[main_name] = self.branches[main_name](h, key_padding_mask=key_padding_mask)

        for stream, out in launched:
            main_stream.wait_stream(stream)
            out.record_stream(main_stream)

        return [outputs[name] for name in self.branch_names]

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        resolved_scheduler, _ = self._resolve_branch_scheduler(
            device=h.device,
            compiled=_is_torch_compiling(),
            dtype=h.dtype,
            batch_size=h.shape[0],
            seqlen=h.shape[1],
        )
        if resolved_scheduler == "cuda_streams":
            branch_outputs = self._run_branches_cuda_streams(h, key_padding_mask=key_padding_mask)
        else:
            branch_outputs = self._run_branches_sequential(h, key_padding_mask=key_padding_mask)
        fused = self.fusion(x, branch_outputs)
        fused = _mask_keep(fused, key_padding_mask)
        x = x + self.resid_dropout(fused)

        h = self.norm2(x)
        h = self.ffn(h)
        h = _mask_keep(h, key_padding_mask)
        x = x + self.resid_dropout(h)
        return x
