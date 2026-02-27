import torch

from nncore.blocks import TransformerBlock
from nncore.models import MoEConfig


def test_transformer_block_moe_forward_with_aux():
    block = TransformerBlock(
        d_model=16,
        num_heads=4,
        ffn_type="moe",
        moe_cfg=MoEConfig(num_experts=4, top_k=2, aux_loss=True),
    )

    x = torch.randn(2, 6, 16)
    y, aux = block(x, return_aux=True)

    assert y.shape == x.shape
    assert "moe/load_balance" in aux
