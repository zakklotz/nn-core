from __future__ import annotations

import torch
from torch.testing import assert_close
import pytest

from nncore.blocks import OFNBlock
from nncore.models import OFNConfig


def _make_config(**overrides) -> OFNConfig:
    cfg = OFNConfig(
        d_model=32,
        n_heads=4,
        max_seq_len=32,
        num_layers=2,
        dropout=0.0,
    )
    for key, value in overrides.items():
        if key.startswith("field__"):
            setattr(cfg.field, key.split("__", 1)[1], value)
        elif key.startswith("local__"):
            setattr(cfg.operators.local, key.split("__", 1)[1], value)
        elif key.startswith("attention__"):
            setattr(cfg.operators.attention, key.split("__", 1)[1], value)
        elif key.startswith("mediator__"):
            setattr(cfg.mediator, key.split("__", 1)[1], value)
        else:
            setattr(cfg, key, value)
    cfg.__post_init__()
    cfg.field.__post_init__()
    cfg.operators.local.__post_init__()
    cfg.operators.attention.__post_init__()
    cfg.mediator.__post_init__()
    return cfg


def _prefix_pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    prefix = torch.randn(2, 5, 32)
    suffix_a = torch.randn(2, 3, 32)
    suffix_b = torch.randn(2, 3, 32)
    return torch.cat([prefix, suffix_a], dim=1), torch.cat([prefix, suffix_b], dim=1)


def test_ofn_block_shape_and_field_shape():
    block = OFNBlock(_make_config())
    x = torch.randn(2, 8, 32)
    y, field = block(x)

    assert y.shape == x.shape
    assert field.shape == (2, 8, 4, 64)
    assert torch.isfinite(y).all()
    assert torch.isfinite(field).all()


@pytest.mark.parametrize("builder", ["ema", "cumsum"])
def test_ofn_block_builders_run(builder):
    block = OFNBlock(_make_config(field__builder=builder))
    x = torch.randn(2, 8, 32)
    y, field = block(x)

    assert y.shape == x.shape
    assert field.shape == (2, 8, 4, 64)


def test_ofn_window_attention_is_causal():
    block = OFNBlock(_make_config())
    x1, x2 = _prefix_pair()
    _, field1 = block(x1)
    _, field2 = block(x2)

    assert_close(field1[:, :5], field2[:, :5], atol=1e-6, rtol=1e-6)


def test_ofn_no_field_keeps_parallel_shapes():
    block = OFNBlock(_make_config(field__enabled=False, field__feedback=False))
    x = torch.randn(2, 7, 32)
    y, field = block(x)

    assert y.shape == x.shape
    assert field.shape == (2, 7, 4, 64)
    assert block.mediator.out_proj.in_features == 256


def test_ofn_cross_attend_reserved_error():
    with pytest.raises(NotImplementedError, match="cross_attend"):
        OFNBlock(_make_config(field__conditioning="cross_attend"))
