from dataclasses import dataclass

import torch


@dataclass
class LayerKV:
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None


class KVCache:
    def __init__(self, num_layers: int):
        self.layers = [LayerKV() for _ in range(num_layers)]

    def reset(self) -> None:
        for layer in self.layers:
            layer.k = None
            layer.v = None

    def get(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        layer = self.layers[layer_idx]
        length = 0 if layer.k is None else layer.k.shape[2]
        return layer.k, layer.v, length

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        layer = self.layers[layer_idx]

        if k_new.shape != v_new.shape:
            raise ValueError("k_new and v_new must have identical shapes.")

        if layer.k is None:
            layer.k = k_new
            layer.v = v_new
            return

        if layer.v is None:
            raise ValueError("Invalid cache state: k exists but v is None.")

        if layer.k.shape[0] != k_new.shape[0] or layer.k.shape[1] != k_new.shape[1] or layer.k.shape[3] != k_new.shape[3]:
            raise ValueError("Cached and new KV tensors must have matching (B, H, Dh).")

        if layer.k.device != k_new.device or layer.v.device != v_new.device:
            raise ValueError("Cached and new KV tensors must be on the same device.")

        if layer.k.dtype != k_new.dtype or layer.v.dtype != v_new.dtype:
            raise ValueError("Cached and new KV tensors must have the same dtype.")

        layer.k = torch.cat([layer.k, k_new], dim=2)
        layer.v = torch.cat([layer.v, v_new], dim=2)
