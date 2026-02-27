import torch
import torch.nn as nn


class Rope(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}.")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)

        cos = torch.repeat_interleave(torch.cos(freqs), 2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(freqs), 2, dim=-1)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = q.shape[-2]
        cos = self.cos[pos_offset:pos_offset + t].to(device=q.device, dtype=q.dtype).view(1, 1, t, -1)
        sin = self.sin[pos_offset:pos_offset + t].to(device=q.device, dtype=q.dtype).view(1, 1, t, -1)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
