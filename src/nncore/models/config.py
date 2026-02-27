from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Dict


@dataclass
class AttentionConfig:
    backend: str = "manual"          # "manual" or "sdpa"
    dropout_p: float = 0.0
    resid_dropout_p: float = 0.0
    scale: float | None = None
    # Custom normalization callable is NOT serializable; leave it out of config.
    # Tajalli can inject it at runtime via model construction kwargs.
    # normalize: Any = None


@dataclass
class BlockConfig:
    norm_style: str = "pre"          # "pre" or "post"
    mlp_dims: list[int] | None = None
    bias: bool = True


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    d_model: int = 512
    num_heads: int = 8
    max_seq_len: int = 2048

    num_encoder_layers: int = 0
    num_decoder_layers: int = 0

    tie_weights: bool = True
    return_hidden: bool = False

    attn: AttentionConfig = field(default_factory=AttentionConfig)
    block: BlockConfig = field(default_factory=BlockConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TransformerConfig":
        # dataclass-friendly nested reconstruction
        attn_d = d.get("attn", {}) or {}
        block_d = d.get("block", {}) or {}

        return TransformerConfig(
            vocab_size=int(d["vocab_size"]),
            d_model=int(d["d_model"]),
            num_heads=int(d["num_heads"]),
            max_seq_len=int(d["max_seq_len"]),
            num_encoder_layers=int(d.get("num_encoder_layers", 0)),
            num_decoder_layers=int(d.get("num_decoder_layers", 0)),
            tie_weights=bool(d.get("tie_weights", True)),
            return_hidden=bool(d.get("return_hidden", False)),
            attn=AttentionConfig(**attn_d),
            block=BlockConfig(**block_d),
        )
