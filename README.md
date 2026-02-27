# nn-core

A modular neural architecture core for building and training modern deep
learning models.

`nn-core` provides reusable building blocks and training infrastructure
for:

-   Transformer models
-   Recursive / weight-shared architectures
-   Mixture-of-Experts models
-   Depth-scalable networks
-   GPU-first training workflows

This repository is designed to serve as a foundational layer for
higher-level experiment and research repositories.

------------------------------------------------------------------------

## Architecture Overview

nn-core provides:

- Modular attention kernels (manual + SDPA)
- Config-driven Transformer construction
- Encoder, Decoder, and Seq2Seq modes
- Minimal training engine with AMP + grad accumulation
- Toy LM CLI harness
- Strict shape validation utilities

Designed as the foundation for higher-level systems.

------------------------------------------------------------------------

## Philosophy

`nn-core` is intentionally minimal and architecture-focused.

It contains:

-   Reusable neural layers and blocks
-   Recursive computation primitives
-   Mixture-of-Experts infrastructure
-   Generic training engine
-   Checkpointing and metrics
-   Smoke tests for validation

It does **not** contain:

-   Dataset-specific experiments
-   Benchmark pipelines
-   Research notebooks
-   Paper-specific implementations

Those belong in experiment repositories that depend on this core.

------------------------------------------------------------------------

## Quickstart

### Create a tiny decoder-only Transformer

```python
from nncore.models import TransformerConfig, BlockConfig, AttentionConfig, Transformer

cfg = TransformerConfig(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    max_seq_len=1024,
    block=BlockConfig(
        attn=AttentionConfig(n_heads=4),
        norm="rmsnorm",
        positional="rope",
        attn_backend="auto",
        ffn_type="mlp",
    ),
)

model = Transformer(cfg)
```

### Decode-time KV cache (incremental generation)

```python
import torch
from nncore.cache import KVCache

model.eval()
cache = KVCache(num_layers=model.cfg.n_layers)

# prefill
prefix = torch.randint(0, cfg.vocab_size, (1, 16))
logits = model(prefix, kv_cache=cache, is_decode=False)

# decode 1 token at a time
next_tok = torch.randint(0, cfg.vocab_size, (1, 1))
logits_step = model(next_tok, kv_cache=cache, is_decode=True)
```

### MoE + aux losses (opt-in)

```python
from nncore.models import MoEConfig

cfg.block.ffn_type = "moe"
cfg.block.moe = MoEConfig(num_experts=8, top_k=2, aux_loss=True)

logits, aux = model(torch.randint(0, cfg.vocab_size, (1, 8)), return_aux=True)
print(aux.keys())
```

### Recurrence (shared block)

```python
cfg.recursive = True
cfg.recurrence_steps = 4
logits = model(torch.randint(0, cfg.vocab_size, (1, 32)))
```


## Installation

Editable install:

``` bash
pip install -e .
```

Development tools:

``` bash
pip install -e .[dev]
```

------------------------------------------------------------------------

## Core Capabilities

### Transformer Primitives

-   Multi-head self-attention
-   Rotary position embeddings (RoPE)
-   Pre-norm transformer blocks
-   Decoder-only language model skeleton
-   KV-cache support (optional)

### Recursive Computation

-   Weight-shared block application
-   Variable depth execution
-   Depth override at inference

### Mixture-of-Experts

-   Top-k routing
-   Load balancing
-   Auxiliary regularization hooks

### Training Infrastructure

-   Mixed precision (AMP)
-   Gradient clipping
-   Optimizer and scheduler factories
-   Checkpoint save/load
-   Metric tracking
-   Automatic device detection (CUDA-first)

------------------------------------------------------------------------

## Smoke Tests

After installation:

``` bash
pytest -q
```

Or run a minimal sanity check:

``` bash
python scripts/smoke_toy_lm.py
```

Smoke tests verify:

-   Forward pass correctness
-   Loss decreases during training
-   Checkpoint round-trip integrity
-   No NaNs under mixed precision

------------------------------------------------------------------------

## Design Principles

1.  GPU-first, CPU-compatible\
2.  Recursion as a first-class primitive\
3.  Clear separation between core and experiments\
4.  No framework lock-in\
5.  Extensible for future architectures

------------------------------------------------------------------------

## Project Structure (High-Level)

    scripts/                 # minimal smoke scripts / entrypoints
    src/nncore/
      layers/                # atomic neural layers (attention, norms, etc.)
      blocks/                # composed modules (transformer blocks)
      models/                # reference models (encoder/decoder/seq2seq)
      functional/            # backend routing (e.g., attention kernels)
      positional/            # positional encodings (RoPE)
      cache/                 # KVCache (decode) + StepCache seam (steps)
      moe/                   # router/experts/MoELayer
      recurrence/            # RecurrenceEngine, UpdateRules, ExitRouter seam
      constraints/           # loss plugins (registry + base protocol)
      hooks/                 # hidden/logits/loss hooks
      train/                 # trainer/engine
      utils/                 # logging, meters, shapes, device helpers

------------------------------------------------------------------------

## License

MIT License --- see `LICENSE` for details.

------------------------------------------------------------------------

## Author

Zachary Klotz
