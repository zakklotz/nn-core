import pytest

from nncore.models import TajalliyatConfig


def test_tajalliyat_config_roundtrip():
    cfg = TajalliyatConfig(
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
        use_attention=True,
        use_cnn=True,
        use_mamba=False,
        fusion_type="gated_sum",
        cnn_kernel_size=5,
        attention_branch_proj=True,
        cnn_branch_proj=True,
        branch_dropout=0.2,
        branch_scheduler="cuda_streams",
    )

    assert TajalliyatConfig.from_dict(cfg.to_dict()) == cfg


def test_tajalliyat_config_requires_one_branch():
    with pytest.raises(ValueError, match="At least one Tajalliyat branch"):
        TajalliyatConfig(use_attention=False, use_cnn=False, use_mamba=False)


def test_tajalliyat_config_rejects_invalid_fusion_type():
    with pytest.raises(ValueError, match="fusion_type"):
        TajalliyatConfig(fusion_type="bad")


def test_tajalliyat_config_rejects_invalid_branch_scheduler():
    with pytest.raises(ValueError, match="branch_scheduler"):
        TajalliyatConfig(branch_scheduler="bad")


def test_tajalliyat_config_rejects_bad_head_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        TajalliyatConfig(d_model=30, n_heads=8)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("cnn_kernel_size", 0),
        ("mamba_d_state", 0),
        ("mamba_d_conv", 0),
        ("mamba_expand", 0),
        ("ffn_mult", 0.0),
        ("ffn_mult", 1e-4),
    ],
)
def test_tajalliyat_config_rejects_nonpositive_sizes(field_name, value):
    kwargs = {field_name: value}
    with pytest.raises(ValueError):
        TajalliyatConfig(**kwargs)
