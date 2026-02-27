import torch
import torch.nn as nn

from nncore.models import Transformer, TransformerConfig
from nncore.recurrence import NullExitRouter, RecurrenceEngine, ResidualRule


class _OnesUpdateBlock(nn.Module):
    def forward(self, h, **kwargs):
        return torch.ones_like(h)


class _FreezeFirstTokenRouter(nn.Module):
    def forward(self, h, step_idx: int, *, state=None):
        mask = torch.zeros(h.shape[0], h.shape[1], dtype=torch.bool, device=h.device)
        if step_idx == 0:
            mask[:, 0] = True
        return mask


def test_exit_router_freezing_behavior():
    block = _OnesUpdateBlock()
    engine = RecurrenceEngine(block=block, rule=ResidualRule(), n_steps_default=3)

    h = torch.zeros(1, 4, 2)
    out = engine(h, exit_router=_FreezeFirstTokenRouter())

    assert torch.allclose(out[:, 0, :], torch.zeros_like(out[:, 0, :]))
    assert torch.allclose(out[:, 1:, :], torch.full_like(out[:, 1:, :], 3.0))


def test_transformer_recursive_accepts_exit_router():
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

    logits = model(x, exit_router=NullExitRouter())
    assert logits.shape == (2, 8, 64)
