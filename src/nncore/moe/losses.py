import torch


def load_balance_loss(probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    num_experts = probs.shape[-1]
    importance = probs.sum(dim=(0, 1))
    importance = importance / importance.sum().clamp_min(1e-12)

    top1 = topk_indices[..., 0]
    load = torch.nn.functional.one_hot(top1, num_classes=num_experts).float().sum(dim=(0, 1))
    load = load / load.sum().clamp_min(1e-12)

    return (importance * load).sum() * (num_experts ** 2)


def router_entropy_loss(probs: torch.Tensor) -> torch.Tensor:
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return -entropy
