import torch.nn as nn


def make_norm(name: str, d_model: int, eps: float) -> nn.Module:
    if name == "layernorm":
        return nn.LayerNorm(d_model, eps=eps)
    if name == "rmsnorm":
        from nncore.layers.norm import RMSNorm

        return RMSNorm(d_model, eps=eps)
    raise ValueError(f"Unknown norm type: {name}")
