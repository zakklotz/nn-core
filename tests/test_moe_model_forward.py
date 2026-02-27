import torch

from nncore.models import MoEConfig, Transformer, TransformerConfig


def test_transformer_forward_return_aux_with_moe():
    cfg = TransformerConfig(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        max_seq_len=16,
        num_encoder_layers=0,
        num_decoder_layers=2,
    )
    cfg.block.ffn_type = "moe"
    cfg.block.moe = MoEConfig(num_experts=4, top_k=2, aux_loss=True)

    model = Transformer(config=cfg)
    model.eval()

    x = torch.randint(0, 64, (2, 8))
    logits, aux = model(x, return_aux=True)

    assert logits.shape == (2, 8, 64)
    assert "moe/load_balance" in aux
