import torch

from nncore.models import TajalliyatConfig, TajalliyatLM
from nncore.smoke import ToyLMConfig, make_toy_lm_batch, toy_lm_forward_fn
from nncore.train import Trainer


def _make_model_config(**overrides) -> TajalliyatConfig:
    cfg = TajalliyatConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        max_seq_len=16,
        num_layers=2,
        dropout=0.0,
        use_attention=True,
        use_cnn=True,
        use_mamba=False,
        fusion_type="gated_sum",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg


def test_tajalliyat_model_config_path_logits():
    cfg = _make_model_config()
    model = TajalliyatLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    logits = model(x)

    assert logits.shape == (2, 10, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_tajalliyat_model_return_hidden():
    cfg = _make_model_config(return_hidden=True)
    model = TajalliyatLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    hidden = model(x)

    assert hidden.shape == (2, 10, cfg.d_model)
    assert torch.isfinite(hidden).all()


def test_tajalliyat_model_branch_scheduler_status():
    cfg = _make_model_config(branch_scheduler="auto")
    model = TajalliyatLM(cfg)

    status = model.branch_scheduler_status(device="cpu", compiled=False)

    assert status["configured"] == "auto"
    assert status["resolved"] == "sequential"
    assert status["active_branches"] == ["attention", "cnn"]


def test_tajalliyat_toy_lm_smoke_cpu_runs():
    torch.manual_seed(0)
    toy_cfg = ToyLMConfig(vocab_size=64, seq_len=16, batch_size=4)
    model_cfg = _make_model_config(
        vocab_size=toy_cfg.vocab_size,
        max_seq_len=toy_cfg.seq_len,
        use_attention=True,
        use_cnn=True,
        use_mamba=False,
        fusion_type="sum",
    )
    model = TajalliyatLM(model_cfg)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, opt, device="cpu", amp=False)

    for _ in range(2):
        batch = make_toy_lm_batch(toy_cfg, device="cpu")
        out = trainer.train_step(toy_lm_forward_fn, batch)
        assert torch.isfinite(torch.tensor(out["loss"]))
