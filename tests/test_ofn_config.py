import pytest

from nncore.models import OFNConfig


def test_ofn_config_roundtrip():
    cfg = OFNConfig(
        vocab_size=123,
        d_model=48,
        n_heads=6,
        max_seq_len=256,
        num_layers=3,
        positional="rope",
        attn_backend="manual",
        norm="rmsnorm",
        norm_eps=1e-6,
        dropout=0.1,
        ffn_mult=3.0,
    )
    data = cfg.to_dict()
    data["field"]["builder"] = "cumsum"
    data["field"]["ema_timescales"] = [4, 8, 16, 32]
    data["mediator"]["mode"] = "gated_sum"
    cfg = OFNConfig.from_dict(data)

    assert OFNConfig.from_dict(cfg.to_dict()) == cfg


def test_ofn_config_requires_one_operator():
    with pytest.raises(ValueError, match="At least one OFN operator branch"):
        OFNConfig.from_dict(
            {
                "operators": {
                    "local": {"enabled": False},
                    "attention": {"enabled": False},
                }
            }
        )


def test_ofn_config_rejects_invalid_conditioning():
    with pytest.raises(ValueError, match="field.conditioning"):
        OFNConfig.from_dict({"field": {"conditioning": "bad"}})


def test_ofn_config_rejects_bad_ema_shape():
    with pytest.raises(ValueError, match="ema_timescales length"):
        OFNConfig.from_dict({"field": {"slots": 3, "ema_timescales": [8, 32, 128, 512]}})
