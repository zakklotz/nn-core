# Changelog

All notable changes to **nn-core** will be documented in this file.

The project follows Semantic Versioning (SemVer).

## Unreleased

### Added
- (placeholder)

### Changed
- (placeholder)

### Fixed
- (placeholder)

## 0.3.0 — Phase 2: Long-context efficiency & extensibility

### Added
- RoPE positional encoding
- RMSNorm option
- Attention backend routing (manual / SDPA / auto)
- Decode-time KV cache (prefill + incremental decode parity tests)
- StepCache seam for step-wise reuse (e.g., recursion depth)
- MoE FFN option (top-k router, aux losses)
- RecurrenceEngine + UpdateRule API
- ExitRouter seam for adaptive halting
- Constraint plugin system (registry + trainer integration)
- Hook system (hidden/logits/loss interception)

### Changed
- Config system extended with opt-in toggles for Phase 2 features

## 0.2.0 — Phase 1: Core infrastructure milestone

### Added
- Clean modular architecture (layers/blocks/models)
- Attention kernels and transformer blocks
- Config system and CLI harness
- Training engine + shape validation utilities
- Test suite and versioned release baseline
