import torch

from nncore.models import Transformer
from nncore.train import Trainer
from nncore.smoke import ToyLMConfig, make_toy_lm_batch, toy_lm_forward_fn


def test_toy_lm_smoke_cpu_runs():
    cfg = ToyLMConfig(vocab_size=64, seq_len=16, batch_size=4)

    model = Transformer(
        vocab_size=cfg.vocab_size,
        d_model=32,
        num_heads=4,
        max_seq_len=cfg.seq_len,
        num_encoder_layers=0,
        num_decoder_layers=2,
        norm_style="pre",
        attn_backend="manual",
        tie_weights=True,
        return_hidden=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, opt, device="cpu", amp=False)

    # A couple steps just to ensure end-to-end wiring works
    for _ in range(3):
        batch = make_toy_lm_batch(cfg, device="cpu")
        out = trainer.train_step(toy_lm_forward_fn, batch)
        assert torch.isfinite(torch.tensor(out["loss"]))
