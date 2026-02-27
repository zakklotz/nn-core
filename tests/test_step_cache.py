from nncore.cache import StepCache
from nncore.models import Transformer
import torch


def test_step_cache_store_load_reset():
    cache = StepCache(num_layers=3)
    cache.set(0, "kv0", (1, 2))

    assert cache.get(0, "kv0") == (1, 2)
    assert cache.get(1, "kv0") is None

    cache.reset()
    assert cache.get(0, "kv0") is None


def test_model_forward_accepts_step_cache():
    model = Transformer(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        max_seq_len=16,
        num_encoder_layers=0,
        num_decoder_layers=2,
    )
    x = torch.randint(0, 64, (2, 8))

    step_cache = StepCache(num_layers=2)
    y = model(x, step_cache=step_cache, step_idx=0)

    assert y.shape == (2, 8, 64)
