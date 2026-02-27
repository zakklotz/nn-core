from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class MoEConfig:
    num_experts: int = 8
    top_k: int = 2
    expert_hidden: int | None = None
    router: str = "topk"
    aux_loss: bool = True
    aux_load_balance: float = 0.01
    aux_entropy: float = 0.0


@dataclass
class TajalliConfig:
    essence_dim: int = 1024
    n_attributes: int = 7
    alpha_init: float = 0.7
    essence_warmup_steps: int = 2000
    gate_entropy_weight: float = 0.01


@dataclass
class ConstraintConfig:
    name: str
    weight: float = 1.0
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class AttentionConfig:
    backend: str = "manual"          # "manual" or "sdpa"
    attn_backend: str = "sdpa"       # "manual", "sdpa", or "auto"
    dropout_p: float = 0.0
    resid_dropout_p: float = 0.0
    scale: float | None = None
    use_kv_cache: bool = False
    # Custom normalization callable is NOT serializable; leave it out of config.
    # Tajalli can inject it at runtime via model construction kwargs.
    # normalize: Any = None


@dataclass
class BlockConfig:
    norm_style: str = "pre"          # "pre" or "post"
    norm: str = "layernorm"          # "layernorm" or "rmsnorm"
    norm_eps: float = 1e-5
    mlp_dims: list[int] | None = None
    ffn_type: str = "mlp"            # "mlp" or "moe"
    moe: MoEConfig | None = None
    bias: bool = True


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    d_model: int = 512
    num_heads: int = 8
    max_seq_len: int = 2048

    num_encoder_layers: int = 0
    num_decoder_layers: int = 0

    positional: str = "absolute"
    use_step_cache: bool = False
    recursive: bool = False
    recurrence_steps: int = 1
    use_exit_router: bool = False
    constraints: list[ConstraintConfig] | None = None
    hooks: bool = False
    tajalli: TajalliConfig | None = None

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

        moe_d = block_d.get("moe")
        if moe_d is not None:
            block_d = dict(block_d)
            block_d["moe"] = MoEConfig(**moe_d)

        constraints_d = d.get("constraints")
        constraints = None
        if constraints_d is not None:
            constraints = [ConstraintConfig(**item) for item in constraints_d]

        tajalli_d = d.get("tajalli")
        tajalli = TajalliConfig(**tajalli_d) if tajalli_d is not None else None

        return TransformerConfig(
            vocab_size=int(d["vocab_size"]),
            d_model=int(d["d_model"]),
            num_heads=int(d["num_heads"]),
            max_seq_len=int(d["max_seq_len"]),
            num_encoder_layers=int(d.get("num_encoder_layers", 0)),
            num_decoder_layers=int(d.get("num_decoder_layers", 0)),
            positional=str(d.get("positional", "absolute")),
            use_step_cache=bool(d.get("use_step_cache", False)),
            recursive=bool(d.get("recursive", False)),
            recurrence_steps=int(d.get("recurrence_steps", 1)),
            use_exit_router=bool(d.get("use_exit_router", False)),
            constraints=constraints,
            hooks=bool(d.get("hooks", False)),
            tajalli=tajalli,
            tie_weights=bool(d.get("tie_weights", True)),
            return_hidden=bool(d.get("return_hidden", False)),
            attn=AttentionConfig(**attn_d),
            block=BlockConfig(**block_d),
        )
