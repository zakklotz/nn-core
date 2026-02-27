import torch

from nncore.models import Transformer


def test_encoder_only_forward():
    m = Transformer(
        vocab_size=100,
        d_model=32,
        num_heads=4,
        max_seq_len=64,
        num_encoder_layers=2,
        num_decoder_layers=0,
    )
    src = torch.randint(0, 100, (2, 10))
    enc = m(src)
    assert enc.shape == (2, 10, 32)


def test_decoder_only_forward_logits():
    m = Transformer(
        vocab_size=100,
        d_model=32,
        num_heads=4,
        max_seq_len=64,
        num_encoder_layers=0,
        num_decoder_layers=2,
        return_hidden=False,
    )
    tgt = torch.randint(0, 100, (2, 11))
    logits = m(tgt)
    assert logits.shape == (2, 11, 100)
    assert torch.isfinite(logits).all()


def test_seq2seq_forward_logits():
    m = Transformer(
        vocab_size=100,
        d_model=32,
        num_heads=4,
        max_seq_len=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        return_hidden=False,
    )
    src = torch.randint(0, 100, (2, 9))
    tgt = torch.randint(0, 100, (2, 7))
    logits = m(src, tgt)
    assert logits.shape == (2, 7, 100)
    assert torch.isfinite(logits).all()
