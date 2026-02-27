import torch

from nncore.cache import KVCache
from nncore.models import Transformer, TransformerConfig


def _run_kv_cache_parity(positional: str):
    torch.manual_seed(0)

    cfg = TransformerConfig(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        max_seq_len=32,
        num_encoder_layers=0,
        num_decoder_layers=2,
    )
    cfg.positional = positional
    cfg.attn.attn_backend = "manual"
    cfg.attn.use_kv_cache = True

    model = Transformer(config=cfg)
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        logits_full = model(ids)

        t0 = 5
        cache = KVCache(num_layers=cfg.num_decoder_layers)
        _ = model(ids[:, :t0], kv_cache=cache, is_decode=False)

        step_logits = []
        for t in range(t0, ids.shape[1]):
            logits_step = model(ids[:, t:t + 1], kv_cache=cache, is_decode=True)
            step_logits.append(logits_step)

        logits_cached_tail = torch.cat(step_logits, dim=1)

    assert logits_cached_tail.shape == logits_full[:, t0:, :].shape
    assert torch.allclose(logits_cached_tail, logits_full[:, t0:, :], atol=1e-5, rtol=1e-4)

    _, _, cache_len = cache.get(0)
    assert cache_len == ids.shape[1]


def test_kv_cache_decode_parity_rope():
    _run_kv_cache_parity("rope")


def test_kv_cache_decode_parity_absolute():
    _run_kv_cache_parity("absolute")
