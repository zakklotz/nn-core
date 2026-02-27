import torch
import torch.nn as nn


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
