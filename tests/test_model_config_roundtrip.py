from nncore.models import TransformerConfig


def test_transformer_config_roundtrip():
    cfg = TransformerConfig(
        vocab_size=123,
        d_model=64,
        num_heads=8,
        max_seq_len=256,
        num_encoder_layers=2,
        num_decoder_layers=3,
        tie_weights=True,
        return_hidden=False,
    )
    cfg.attn.backend = "manual"
    cfg.attn.dropout_p = 0.1
    cfg.block.norm_style = "pre"
    cfg.block.mlp_dims = [64, 256, 64]

    d = cfg.to_dict()
    cfg2 = TransformerConfig.from_dict(d)

    assert cfg2.vocab_size == cfg.vocab_size
    assert cfg2.d_model == cfg.d_model
    assert cfg2.num_heads == cfg.num_heads
    assert cfg2.max_seq_len == cfg.max_seq_len
    assert cfg2.num_encoder_layers == cfg.num_encoder_layers
    assert cfg2.num_decoder_layers == cfg.num_decoder_layers
    assert cfg2.attn.backend == cfg.attn.backend
    assert abs(cfg2.attn.dropout_p - cfg.attn.dropout_p) < 1e-9
    assert cfg2.block.norm_style == cfg.block.norm_style
    assert cfg2.block.mlp_dims == cfg.block.mlp_dims
