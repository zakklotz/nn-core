# TajalliyatBlock

`TajalliyatBlock` is a decoder-only language-model block for `nn-core` that runs multiple causal branches over the same token stream and fuses their token-aligned outputs.

## Branches

- `AttentionBranch`: standard causal self-attention for long-range token mixing.
- `CNNBranch`: causal 1D convolution with explicit left padding for local pattern extraction.
- `MambaBranch`: official `mamba_ssm.Mamba2` branch for causal state-space sequence mixing.

Each active branch receives the same hidden state tensor `[B, T, D]` and returns `[B, T, D]`.

## Fusion

- `sum`: direct sum of active branch outputs.
- `gated_sum`: tokenwise softmax over active branches from a small gating MLP.
- `barzakh`: mediator fusion using the original token state and every branch output.

`BarzakhFusion` uses:

- `C = concat(x, b1, ..., bN)` with shape `[B, T, (N+1)D]`
- `m = mediator_mlp(C)` with shape `[B, T, D]`
- `a = softmax(branch_logits(m), dim=-1)` with shape `[B, T, N]`
- `u_i = branch_proj_i(b_i)` with shape `[B, T, D]`
- `fused = mediator_out(m) + sum_i a_i * u_i`

## Recommended Ablation Order

1. attention only
2. attention + cnn
3. attention + cnn + mamba
4. compare `sum`, `gated_sum`, and `barzakh`

## Fairness Note

`barzakh` adds substantially more fusion parameters than `sum` or `gated_sum`, and enabling more branches also increases total block capacity because each branch runs at full `d_model` width. For Phase 2, report parameter counts for every ablation and do not interpret fusion wins as architecture-only wins without that context.
