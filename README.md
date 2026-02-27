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
python scripts/smoke_lm.py
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

    src/nncore/
      layers/        # atomic neural layers
      blocks/        # composed modules (attention, transformer, MoE)
      models/        # reference skeletons
      train/         # generic training engine
      optim/         # optimizer + schedulers
      losses/        # reusable losses
      smoke/         # minimal end-to-end verification tasks

------------------------------------------------------------------------

## License

MIT License --- see `LICENSE` for details.

------------------------------------------------------------------------

## Author

Zachary Klotz
