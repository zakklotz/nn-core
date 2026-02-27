import torch
import torch.nn as nn

from nncore.models.config import MoEConfig
from nncore.moe.experts import ExpertFFN
from nncore.moe.losses import load_balance_loss, router_entropy_loss
from nncore.moe.router import TopKRouter


class MoELayer(nn.Module):
    def __init__(self, d_model: int, cfg: MoEConfig, mlp_hidden_fallback: int):
        super().__init__()
        expert_hidden = cfg.expert_hidden if cfg.expert_hidden is not None else mlp_hidden_fallback
        self.cfg = cfg
        self.router = TopKRouter(d_model=d_model, num_experts=cfg.num_experts, top_k=cfg.top_k)
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model=d_model, d_hidden=expert_hidden) for _ in range(cfg.num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        indices, weights, probs = self.router(x)
        out = torch.zeros_like(x)

        for expert_idx, expert in enumerate(self.experts):
            mask = indices == expert_idx
            for k_slot in range(self.cfg.top_k):
                slot_mask = mask[..., k_slot]
                if not slot_mask.any():
                    continue
                x_tokens = x[slot_mask]
                y_tokens = expert(x_tokens)
                w = weights[..., k_slot][slot_mask].unsqueeze(-1)
                out[slot_mask] = out[slot_mask] + (w * y_tokens)

        aux_losses: dict[str, torch.Tensor] = {}
        if self.cfg.aux_loss:
            aux_losses["moe/load_balance"] = self.cfg.aux_load_balance * load_balance_loss(probs, indices)
            if self.cfg.aux_entropy != 0.0:
                aux_losses["moe/router_entropy"] = self.cfg.aux_entropy * router_entropy_loss(probs)

        return out, aux_losses
