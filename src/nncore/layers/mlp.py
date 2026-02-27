import torch
import torch.nn as nn
from typing import Optional, Sequence


class MLP(nn.Module):
    """
    A simple feed-forward MLP defined by a list of layer dimensions.

    dimensions: [d0, d1, ..., dn]
      creates Linear(d0->d1), Linear(d1->d2), ..., Linear(d(n-1)->dn)

    activations:
      - None: uses GELU on hidden layers only (no activation after last layer)
      - len == (len(dimensions) - 2): activations for hidden layers only
      - len == (len(dimensions) - 1): activations for all layers (including last)
    """
    def __init__(
        self,
        dimensions: Sequence[int],
        activations: Optional[Sequence[nn.Module]] = None,
    ):
        super().__init__()
        if len(dimensions) < 2:
            raise ValueError("dimensions must have at least 2 entries (in_dim, out_dim).")

        self.layers = nn.ModuleList(
            [nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(len(dimensions) - 1)]
        )

        num_layers = len(self.layers)
        num_hidden = max(0, num_layers - 1)

        if activations is None:
            # Default: GELU on hidden layers, Identity on last
            acts = [nn.GELU() for _ in range(num_hidden)] + ([nn.Identity()] if num_layers > 0 else [])
        else:
            acts = list(activations)  # copy, do not mutate caller
            if len(acts) == num_hidden:
                # Provided only for hidden layers; add Identity for last layer
                acts = acts + ([nn.Identity()] if num_layers > 0 else [])
            elif len(acts) != num_layers:
                raise ValueError(
                    f"activations must be None, length {num_hidden} (hidden-only), "
                    f"or length {num_layers} (per-layer). Got {len(acts)}."
                )

        # Store as ModuleList so modules are registered properly
        self.activations = nn.ModuleList(acts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x
