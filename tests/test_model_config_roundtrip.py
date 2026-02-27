from nncore.models import ConstraintConfig, MoEConfig, TransformerConfig


def test_transformer_config_roundtrip():
    cfg = TransformerConfig(
        vocab_size=123,
        d_model=64,
        num_heads=8,
        max_seq_len=256,
        num_encoder_layers=2,
        num_decoder_layers=3,
        positional="rope",
        use_step_cache=True,
        recursive=True,
        recurrence_steps=2,
        use_exit_router=True,
        constraints=[ConstraintConfig(name="latency", weight=0.2, params={"target_ms": 12})],
        hooks=True,
        tie_weights=True,
        return_hidden=False,
    )
    cfg.attn.backend = "manual"
    cfg.attn.attn_backend = "auto"
    cfg.attn.dropout_p = 0.1
    cfg.attn.use_kv_cache = True
    cfg.block.norm_style = "pre"
    cfg.block.norm = "rmsnorm"
    cfg.block.norm_eps = 1e-6
    cfg.block.mlp_dims = [64, 256, 64]
    cfg.block.ffn_type = "moe"
    cfg.block.moe = MoEConfig(num_experts=4, top_k=1)

    d = cfg.to_dict()
    cfg2 = TransformerConfig.from_dict(d)

    assert cfg2.vocab_size == cfg.vocab_size
    assert cfg2.d_model == cfg.d_model
    assert cfg2.num_heads == cfg.num_heads
    assert cfg2.max_seq_len == cfg.max_seq_len
    assert cfg2.num_encoder_layers == cfg.num_encoder_layers
    assert cfg2.num_decoder_layers == cfg.num_decoder_layers
    assert cfg2.positional == cfg.positional
    assert cfg2.use_step_cache == cfg.use_step_cache
    assert cfg2.recursive == cfg.recursive
    assert cfg2.recurrence_steps == cfg.recurrence_steps
    assert cfg2.use_exit_router == cfg.use_exit_router
    assert cfg2.constraints is not None
    assert cfg2.constraints[0].name == "latency"
    assert cfg2.constraints[0].params["target_ms"] == 12
    assert cfg2.hooks == cfg.hooks
    assert cfg2.attn.backend == cfg.attn.backend
    assert cfg2.attn.attn_backend == cfg.attn.attn_backend
    assert abs(cfg2.attn.dropout_p - cfg.attn.dropout_p) < 1e-9
    assert cfg2.attn.use_kv_cache == cfg.attn.use_kv_cache
    assert cfg2.block.norm_style == cfg.block.norm_style
    assert cfg2.block.norm == cfg.block.norm
    assert cfg2.block.norm_eps == cfg.block.norm_eps
    assert cfg2.block.mlp_dims == cfg.block.mlp_dims
    assert cfg2.block.ffn_type == cfg.block.ffn_type
    assert cfg2.block.moe is not None
    assert cfg2.block.moe.num_experts == 4


def test_transformer_config_new_defaults():
    cfg = TransformerConfig()

    assert cfg.positional == "absolute"
    assert cfg.block.norm == "layernorm"
    assert cfg.block.norm_eps == 1e-5
    assert cfg.attn.attn_backend == "sdpa"
    assert cfg.attn.use_kv_cache is False
    assert cfg.use_step_cache is False
    assert cfg.block.ffn_type == "mlp"
    assert cfg.block.moe is None
    assert cfg.recursive is False
    assert cfg.recurrence_steps == 1
    assert cfg.use_exit_router is False
    assert cfg.constraints is None
    assert cfg.hooks is False
