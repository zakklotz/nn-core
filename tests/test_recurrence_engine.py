import torch
import torch.nn as nn

from nncore.models import MoEConfig, Transformer, TransformerConfig
from nncore.recurrence import RecurrenceEngine, ResidualRule


class _OnesUpdateBlock(nn.Module):
    def forward(self, h, **kwargs):
        return torch.ones_like(h)


def test_recurrence_engine_basic_shape():
    block = _OnesUpdateBlock()
    engine = RecurrenceEngine(block=block, rule=ResidualRule(), n_steps_default=3)

    h = torch.randn(2, 5, 8)
    out = engine(h)
    assert out.shape == h.shape


def test_recurrence_engine_residual_update_count():
    block = _OnesUpdateBlock()
    engine = RecurrenceEngine(block=block, rule=ResidualRule(), n_steps_default=3)

    h = torch.zeros(1, 2, 4)
    out = engine(h)
    expected = h + 3.0 * torch.ones_like(h)
    assert torch.allclose(out, expected)


def test_transformer_recursive_forward_runs():
    cfg = TransformerConfig(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        max_seq_len=16,
        num_encoder_layers=0,
        num_decoder_layers=2,
    )
    cfg.recursive = True
    cfg.recurrence_steps = 2

    model = Transformer(config=cfg)
    x = torch.randint(0, 64, (2, 8))
    logits = model(x)

    assert logits.shape == (2, 8, 64)


def test_transformer_recursive_forward_aux_with_moe():
    cfg = TransformerConfig(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        max_seq_len=16,
        num_encoder_layers=0,
        num_decoder_layers=2,
    )
    cfg.recursive = True
    cfg.recurrence_steps = 2
    cfg.block.ffn_type = "moe"
    cfg.block.moe = MoEConfig(num_experts=4, top_k=2, aux_loss=True)

    model = Transformer(config=cfg)
    model.eval()

    x = torch.randint(0, 64, (1, 8))
    logits, aux = model(x, return_aux=True)

    assert logits.shape == (1, 8, 64)
    assert "moe/load_balance" in aux
