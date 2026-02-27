import torch
import torch.nn as nn


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.proj = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.proj(x)
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k=self.top_k, dim=-1)
        return indices, weights, probs
