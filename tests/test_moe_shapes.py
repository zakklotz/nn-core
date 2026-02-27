import torch

from nncore.models import MoEConfig
from nncore.moe import MoELayer


def test_moe_layer_output_shape_and_aux_dict():
    cfg = MoEConfig(num_experts=4, top_k=2)
    moe = MoELayer(d_model=16, cfg=cfg, mlp_hidden_fallback=32)

    x = torch.randn(2, 5, 16)
    y, aux = moe(x)

    assert y.shape == x.shape
    assert isinstance(aux, dict)
