import pytest
import torch

from nncore.models import Transformer, TransformerConfig
from nncore.positional import Rope


def test_rope_apply_shape():
    rope = Rope(dim=64, max_seq_len=128)
    q = torch.randn(2, 4, 10, 64)
    k = torch.randn(2, 4, 10, 64)

    q_rot, k_rot = rope.apply(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_dim_must_be_even():
    with pytest.raises(ValueError):
        Rope(dim=63, max_seq_len=128)


def test_transformer_forward_with_rope_positional():
    cfg = TransformerConfig(
        vocab_size=100,
        d_model=32,
        num_heads=4,
        max_seq_len=64,
        num_encoder_layers=0,
        num_decoder_layers=2,
    )
    cfg.positional = "rope"

    model = Transformer(config=cfg, return_hidden=False)
    tgt = torch.randint(0, 100, (2, 11))
    logits = model(tgt)

    assert logits.shape == (2, 11, 100)
    assert torch.isfinite(logits).all()
