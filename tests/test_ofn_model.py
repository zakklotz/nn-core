from __future__ import annotations

import torch
from torch.testing import assert_close

from nncore.models import OFNConfig, OFNLM


def test_ofn_model_logits_shape():
    cfg = OFNConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        max_seq_len=16,
        num_layers=2,
        positional="rope",
        dropout=0.0,
    )
    model = OFNLM(cfg)
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)

    logits = model(input_ids)
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()


def test_ofn_model_is_causal_under_suffix_perturbation():
    cfg = OFNConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        max_seq_len=16,
        num_layers=2,
        positional="rope",
        dropout=0.0,
    )
    model = OFNLM(cfg)
    torch.manual_seed(0)
    prefix = torch.randint(0, 64, (2, 5), dtype=torch.long)
    suffix_a = torch.randint(0, 64, (2, 3), dtype=torch.long)
    suffix_b = torch.randint(0, 64, (2, 3), dtype=torch.long)

    logits_a = model(torch.cat([prefix, suffix_a], dim=1))
    logits_b = model(torch.cat([prefix, suffix_b], dim=1))

    assert_close(logits_a[:, :5], logits_b[:, :5], atol=1e-6, rtol=1e-6)
