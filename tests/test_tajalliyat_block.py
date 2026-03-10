from __future__ import annotations

import importlib.util

import pytest
import torch
from torch.testing import assert_close

from nncore.blocks import AttentionBranch, CNNBranch, MambaBranch, TajalliyatBlock
from nncore.models import TajalliyatConfig


MAMBA_AVAILABLE = importlib.util.find_spec("mamba_ssm") is not None


def _make_config(**overrides) -> TajalliyatConfig:
    cfg = TajalliyatConfig(
        d_model=32,
        n_heads=4,
        max_seq_len=32,
        num_layers=2,
        dropout=0.0,
        branch_dropout=0.0,
        use_attention=True,
        use_cnn=False,
        use_mamba=False,
        fusion_type="sum",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg


def _prefix_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    prefix = torch.randn(2, 5, 32)
    suffix_a = torch.randn(2, 3, 32)
    suffix_b = torch.randn(2, 3, 32)
    return torch.cat([prefix, suffix_a], dim=1), torch.cat([prefix, suffix_b], dim=1)


def test_attention_only_block_shape():
    block = TajalliyatBlock(_make_config(use_attention=True, use_cnn=False))
    x = torch.randn(2, 8, 32)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_attention_cnn_block_shape():
    block = TajalliyatBlock(_make_config(use_attention=True, use_cnn=True, cnn_kernel_size=5))
    x = torch.randn(2, 8, 32)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_cnn_only_block_shape():
    block = TajalliyatBlock(_make_config(use_attention=False, use_cnn=True))
    x = torch.randn(2, 8, 32)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("fusion_type", ["sum", "gated_sum", "barzakh"])
def test_fusion_modes_produce_finite_outputs(fusion_type):
    block = TajalliyatBlock(
        _make_config(
            use_attention=True,
            use_cnn=True,
            fusion_type=fusion_type,
        )
    )
    x = torch.randn(2, 7, 32)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("fusion_type", ["sum", "gated_sum", "barzakh"])
def test_single_branch_fusions_work(fusion_type):
    block = TajalliyatBlock(
        _make_config(use_attention=True, use_cnn=False, use_mamba=False, fusion_type=fusion_type)
    )
    x = torch.randn(2, 7, 32)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_barzakh_fusion_has_no_nans():
    block = TajalliyatBlock(
        _make_config(use_attention=True, use_cnn=True, fusion_type="barzakh")
    )
    x = torch.randn(2, 9, 32)
    y = block(x)
    assert not torch.isnan(y).any()


def test_attention_branch_is_causal():
    branch = AttentionBranch(32, 4, attn_backend="manual", dropout=0.0)
    x1, x2 = _prefix_pair()
    y1 = branch(x1)
    y2 = branch(x2)

    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6, rtol=1e-6)


def test_cnn_branch_is_causal():
    branch = CNNBranch(32, kernel_size=5)
    x1, x2 = _prefix_pair()
    y1 = branch(x1)
    y2 = branch(x2)

    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6, rtol=1e-6)


def test_mamba_branch_missing_dependency_error():
    if MAMBA_AVAILABLE:
        pytest.skip("mamba_ssm is installed; missing dependency path is not applicable.")

    with pytest.raises(ImportError, match="mamba-ssm is required when use_mamba=True"):
        MambaBranch(32, d_state=64, d_conv=4, expand=2)


def test_disabled_branches_are_not_instantiated():
    block = TajalliyatBlock(_make_config(use_attention=False, use_cnn=True, use_mamba=False))
    assert list(block.branches.keys()) == ["cnn"]


def test_auto_scheduler_falls_back_to_sequential_for_single_branch():
    block = TajalliyatBlock(_make_config(use_attention=True, use_cnn=False, branch_scheduler="auto"))
    status = block.branch_scheduler_status(device="cpu", compiled=False)

    assert status["configured"] == "auto"
    assert status["resolved"] == "sequential"
    assert status["fallback_reason"] == "single_active_branch"


def test_explicit_cuda_streams_requires_cuda():
    block = TajalliyatBlock(
        _make_config(use_attention=True, use_cnn=True, use_mamba=False, branch_scheduler="cuda_streams")
    )
    x = torch.randn(2, 8, 32)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        block(x)


def test_auto_scheduler_reports_compiled_fallback(monkeypatch: pytest.MonkeyPatch):
    block = TajalliyatBlock(_make_config(use_attention=True, use_cnn=True, branch_scheduler="auto"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    status = block.branch_scheduler_status(device="cuda", compiled=True)

    assert status["resolved"] == "sequential"
    assert status["fallback_reason"] == "compiled_model"


def test_auto_scheduler_uses_autotuned_choice(monkeypatch: pytest.MonkeyPatch):
    block = TajalliyatBlock(_make_config(use_attention=True, use_cnn=True, branch_scheduler="auto"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        block,
        "_select_auto_scheduler",
        lambda **kwargs: ("sequential", "auto_benchmark_selected_sequential"),
    )

    status = block.branch_scheduler_status(device="cuda", compiled=False)

    assert status["resolved"] == "sequential"
    assert status["fallback_reason"] == "auto_benchmark_selected_sequential"


def test_auto_scheduler_can_select_cuda_streams(monkeypatch: pytest.MonkeyPatch):
    block = TajalliyatBlock(_make_config(use_attention=True, use_cnn=True, branch_scheduler="auto"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(block, "_select_auto_scheduler", lambda **kwargs: ("cuda_streams", None))

    status = block.branch_scheduler_status(device="cuda", compiled=False)

    assert status["resolved"] == "cuda_streams"
    assert status["fallback_reason"] is None


def test_gated_and_barzakh_are_distinct_fusions():
    gated = TajalliyatBlock(_make_config(use_attention=True, use_cnn=True, fusion_type="gated_sum"))
    barzakh = TajalliyatBlock(_make_config(use_attention=True, use_cnn=True, fusion_type="barzakh"))

    gated_params = sum(p.numel() for p in gated.fusion.parameters())
    barzakh_params = sum(p.numel() for p in barzakh.fusion.parameters())

    assert gated.fusion.__class__.__name__ == "GatedFusion"
    assert barzakh.fusion.__class__.__name__ == "BarzakhFusion"
    assert barzakh_params > gated_params


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA scheduler parity test requires CUDA.",
)
def test_attention_cnn_cuda_streams_matches_sequential():
    torch.manual_seed(0)
    seq_block = TajalliyatBlock(
        _make_config(
            use_attention=True,
            use_cnn=True,
            use_mamba=False,
            fusion_type="sum",
            branch_scheduler="sequential",
        )
    ).to("cuda")
    stream_block = TajalliyatBlock(
        _make_config(
            use_attention=True,
            use_cnn=True,
            use_mamba=False,
            fusion_type="sum",
            branch_scheduler="cuda_streams",
        )
    ).to("cuda")
    stream_block.load_state_dict(seq_block.state_dict())

    x_seq = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
    x_stream = x_seq.detach().clone().requires_grad_(True)
    key_padding_mask = torch.tensor(
        [
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, True, True, True, True],
        ],
        dtype=torch.bool,
        device="cuda",
    )

    y_seq = seq_block(x_seq, key_padding_mask=key_padding_mask)
    y_stream = stream_block(x_stream, key_padding_mask=key_padding_mask)

    assert_close(y_seq, y_stream, atol=1e-5, rtol=1e-5)

    y_seq.square().mean().backward()
    y_stream.square().mean().backward()

    assert_close(x_seq.grad, x_stream.grad, atol=1e-5, rtol=1e-5)
    seq_grads = dict(seq_block.named_parameters())
    stream_grads = dict(stream_block.named_parameters())
    for name in seq_grads:
        assert_close(seq_grads[name].grad, stream_grads[name].grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not MAMBA_AVAILABLE or not torch.cuda.is_available(),
    reason="Mamba2 CUDA stream test requires mamba_ssm and CUDA.",
)
def test_attention_cnn_mamba_block_shape_cuda_streams():
    block = TajalliyatBlock(
        _make_config(
            use_attention=True,
            use_cnn=True,
            use_mamba=True,
            fusion_type="barzakh",
            branch_scheduler="cuda_streams",
        )
    ).to("cuda")
    x = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
