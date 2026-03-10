from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
from typing import Any, Dict, Literal


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
class ConstraintConfig:
    name: str
    weight: float = 1.0
    params: dict[str, object] = dataclass_field(default_factory=dict)


@dataclass
class AttentionConfig:
    backend: str = "manual"          # "manual" or "sdpa"
    attn_backend: str = "sdpa"       # "manual", "sdpa", or "auto"
    dropout_p: float = 0.0
    resid_dropout_p: float = 0.0
    scale: float | None = None
    use_kv_cache: bool = False
    # Custom normalization callable is NOT serializable; leave it out of config.
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

    tie_weights: bool = True
    return_hidden: bool = False

    attn: AttentionConfig = dataclass_field(default_factory=AttentionConfig)
    block: BlockConfig = dataclass_field(default_factory=BlockConfig)

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
            tie_weights=bool(d.get("tie_weights", True)),
            return_hidden=bool(d.get("return_hidden", False)),
            attn=AttentionConfig(**attn_d),
            block=BlockConfig(**block_d),
        )


@dataclass
class TajalliyatConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    max_seq_len: int = 2048
    num_layers: int = 6

    tie_weights: bool = True
    return_hidden: bool = False
    positional: str = "absolute"
    attn_backend: str = "sdpa"
    norm: str = "layernorm"
    norm_eps: float = 1e-5

    dropout: float = 0.0
    ffn_mult: float = 4.0
    use_attention: bool = True
    use_cnn: bool = False
    use_mamba: bool = False
    fusion_type: str = "sum"
    cnn_kernel_size: int = 3
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    attention_branch_proj: bool = False
    cnn_branch_proj: bool = False
    mamba_branch_proj: bool = False
    branch_dropout: float = 0.0
    branch_scheduler: Literal["auto", "sequential", "cuda_streams"] = "auto"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0.")
        if self.positional not in {"absolute", "rope"}:
            raise ValueError("positional must be 'absolute' or 'rope'.")
        if self.norm not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm must be 'layernorm' or 'rmsnorm'.")
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be > 0.")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be in [0, 1].")
        if not (0.0 <= self.branch_dropout <= 1.0):
            raise ValueError("branch_dropout must be in [0, 1].")
        if self.ffn_mult <= 0.0:
            raise ValueError("ffn_mult must be > 0.")
        if int(self.ffn_mult * self.d_model) < 1:
            raise ValueError("int(ffn_mult * d_model) must be >= 1.")
        if not any((self.use_attention, self.use_cnn, self.use_mamba)):
            raise ValueError("At least one Tajalliyat branch must be active.")
        if self.fusion_type not in {"sum", "gated_sum", "barzakh"}:
            raise ValueError("fusion_type must be 'sum', 'gated_sum', or 'barzakh'.")
        if self.cnn_kernel_size <= 0:
            raise ValueError("cnn_kernel_size must be > 0.")
        if self.mamba_d_state <= 0:
            raise ValueError("mamba_d_state must be > 0.")
        if self.mamba_d_conv <= 0:
            raise ValueError("mamba_d_conv must be > 0.")
        if self.mamba_expand <= 0:
            raise ValueError("mamba_expand must be > 0.")
        if self.branch_scheduler not in {"auto", "sequential", "cuda_streams"}:
            raise ValueError("branch_scheduler must be 'auto', 'sequential', or 'cuda_streams'.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TajalliyatConfig":
        return TajalliyatConfig(
            vocab_size=int(d.get("vocab_size", 32000)),
            d_model=int(d.get("d_model", 512)),
            n_heads=int(d.get("n_heads", 8)),
            max_seq_len=int(d.get("max_seq_len", 2048)),
            num_layers=int(d.get("num_layers", 6)),
            tie_weights=bool(d.get("tie_weights", True)),
            return_hidden=bool(d.get("return_hidden", False)),
            positional=str(d.get("positional", "absolute")),
            attn_backend=str(d.get("attn_backend", "sdpa")),
            norm=str(d.get("norm", "layernorm")),
            norm_eps=float(d.get("norm_eps", 1e-5)),
            dropout=float(d.get("dropout", 0.0)),
            ffn_mult=float(d.get("ffn_mult", 4.0)),
            use_attention=bool(d.get("use_attention", True)),
            use_cnn=bool(d.get("use_cnn", False)),
            use_mamba=bool(d.get("use_mamba", False)),
            fusion_type=str(d.get("fusion_type", "sum")),
            cnn_kernel_size=int(d.get("cnn_kernel_size", 3)),
            mamba_d_state=int(d.get("mamba_d_state", 64)),
            mamba_d_conv=int(d.get("mamba_d_conv", 4)),
            mamba_expand=int(d.get("mamba_expand", 2)),
            attention_branch_proj=bool(d.get("attention_branch_proj", False)),
            cnn_branch_proj=bool(d.get("cnn_branch_proj", False)),
            mamba_branch_proj=bool(d.get("mamba_branch_proj", False)),
            branch_dropout=float(d.get("branch_dropout", 0.0)),
            branch_scheduler=str(d.get("branch_scheduler", "auto")),
        )


@dataclass
class OFNFieldConfig:
    enabled: bool = True
    slots: int = 4
    d_field: int = 64
    builder: Literal["ema", "cumsum"] = "ema"
    ema_timescales: list[int] = dataclass_field(default_factory=lambda: [8, 32, 128, 512])
    conditioning: Literal["film", "cross_attend"] = "film"
    feedback: bool = True
    feedback_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.slots <= 0:
            raise ValueError("field.slots must be > 0.")
        if self.d_field <= 0:
            raise ValueError("field.d_field must be > 0.")
        if self.builder not in {"ema", "cumsum"}:
            raise ValueError("field.builder must be 'ema' or 'cumsum'.")
        if self.conditioning not in {"film", "cross_attend"}:
            raise ValueError("field.conditioning must be 'film' or 'cross_attend'.")
        if self.feedback_scale <= 0.0:
            raise ValueError("field.feedback_scale must be > 0.")
        if not self.ema_timescales:
            raise ValueError("field.ema_timescales must not be empty.")
        if len(self.ema_timescales) != self.slots:
            raise ValueError("field.ema_timescales length must match field.slots.")
        if any(int(timescale) <= 0 for timescale in self.ema_timescales):
            raise ValueError("field.ema_timescales values must be > 0.")


@dataclass
class OFNLocalOperatorConfig:
    enabled: bool = True
    d_hidden: int = 768
    kernel_size: int = 5

    def __post_init__(self) -> None:
        if self.d_hidden <= 0:
            raise ValueError("operators.local.d_hidden must be > 0.")
        if self.kernel_size <= 0:
            raise ValueError("operators.local.kernel_size must be > 0.")


@dataclass
class OFNAttentionOperatorConfig:
    enabled: bool = True
    mode: Literal["window", "full"] = "window"
    window_size: int = 256

    def __post_init__(self) -> None:
        if self.mode not in {"window", "full"}:
            raise ValueError("operators.attention.mode must be 'window' or 'full'.")
        if self.window_size <= 0:
            raise ValueError("operators.attention.window_size must be > 0.")


@dataclass
class OFNOperatorConfig:
    local: OFNLocalOperatorConfig = dataclass_field(default_factory=OFNLocalOperatorConfig)
    attention: OFNAttentionOperatorConfig = dataclass_field(default_factory=OFNAttentionOperatorConfig)


@dataclass
class OFNMediatorConfig:
    mode: Literal["barzakh", "gated_sum"] = "barzakh"
    d_imaginal: int = 256
    gate_hidden: int = 256

    def __post_init__(self) -> None:
        if self.mode not in {"barzakh", "gated_sum"}:
            raise ValueError("mediator.mode must be 'barzakh' or 'gated_sum'.")
        if self.d_imaginal <= 0:
            raise ValueError("mediator.d_imaginal must be > 0.")
        if self.gate_hidden <= 0:
            raise ValueError("mediator.gate_hidden must be > 0.")


@dataclass
class OFNConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    max_seq_len: int = 2048
    num_layers: int = 6

    tie_weights: bool = True
    return_hidden: bool = False
    positional: str = "absolute"
    attn_backend: str = "sdpa"
    norm: str = "layernorm"
    norm_eps: float = 1e-5

    dropout: float = 0.0
    ffn_mult: float = 4.0
    field: OFNFieldConfig = dataclass_field(default_factory=OFNFieldConfig)
    operators: OFNOperatorConfig = dataclass_field(default_factory=OFNOperatorConfig)
    mediator: OFNMediatorConfig = dataclass_field(default_factory=OFNMediatorConfig)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0.")
        if self.positional not in {"absolute", "rope"}:
            raise ValueError("positional must be 'absolute' or 'rope'.")
        if self.norm not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm must be 'layernorm' or 'rmsnorm'.")
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be > 0.")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be in [0, 1].")
        if self.ffn_mult <= 0.0:
            raise ValueError("ffn_mult must be > 0.")
        if int(self.ffn_mult * self.d_model) < 1:
            raise ValueError("int(ffn_mult * d_model) must be >= 1.")
        if not (self.operators.local.enabled or self.operators.attention.enabled):
            raise ValueError("At least one OFN operator branch must be active.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OFNConfig":
        field_d = dict(d.get("field", {}) or {})
        operators_d = dict(d.get("operators", {}) or {})
        local_d = dict(operators_d.get("local", {}) or {})
        attention_d = dict(operators_d.get("attention", {}) or {})
        mediator_d = dict(d.get("mediator", {}) or {})
        return OFNConfig(
            vocab_size=int(d.get("vocab_size", 32000)),
            d_model=int(d.get("d_model", 512)),
            n_heads=int(d.get("n_heads", 8)),
            max_seq_len=int(d.get("max_seq_len", 2048)),
            num_layers=int(d.get("num_layers", 6)),
            tie_weights=bool(d.get("tie_weights", True)),
            return_hidden=bool(d.get("return_hidden", False)),
            positional=str(d.get("positional", "absolute")),
            attn_backend=str(d.get("attn_backend", "sdpa")),
            norm=str(d.get("norm", "layernorm")),
            norm_eps=float(d.get("norm_eps", 1e-5)),
            dropout=float(d.get("dropout", 0.0)),
            ffn_mult=float(d.get("ffn_mult", 4.0)),
            field=OFNFieldConfig(
                enabled=bool(field_d.get("enabled", True)),
                slots=int(field_d.get("slots", 4)),
                d_field=int(field_d.get("d_field", 64)),
                builder=str(field_d.get("builder", "ema")),
                ema_timescales=[int(value) for value in field_d.get("ema_timescales", [8, 32, 128, 512])],
                conditioning=str(field_d.get("conditioning", "film")),
                feedback=bool(field_d.get("feedback", True)),
                feedback_scale=float(field_d.get("feedback_scale", 1.0)),
            ),
            operators=OFNOperatorConfig(
                local=OFNLocalOperatorConfig(
                    enabled=bool(local_d.get("enabled", True)),
                    d_hidden=int(local_d.get("d_hidden", 768)),
                    kernel_size=int(local_d.get("kernel_size", 5)),
                ),
                attention=OFNAttentionOperatorConfig(
                    enabled=bool(attention_d.get("enabled", True)),
                    mode=str(attention_d.get("mode", "window")),
                    window_size=int(attention_d.get("window_size", 256)),
                ),
            ),
            mediator=OFNMediatorConfig(
                mode=str(mediator_d.get("mode", "barzakh")),
                d_imaginal=int(mediator_d.get("d_imaginal", 256)),
                gate_hidden=int(mediator_d.get("gate_hidden", 256)),
            ),
        )
