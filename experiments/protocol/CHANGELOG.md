# Experimental protocol changelog

Frozen protocol files are immutable. A scientific change requires a copied file
with a new semantic version and a dated entry here. Path-only deployment values,
such as dataset and output roots, do not change the protocol version.

## Inference amendment 1.1.0 — development

- Specify per-image auto-letterboxing directly from decoded frames, independent
  of batch composition; corrected extraction goes to `validation_candidates/v2`.
- Define frame-weighted EDTC as the frame-count-weighted mean of sequence
  scores. Preserve earlier pooled-component outputs as diagnostics.
- Preserve completed 1.0.0 extraction/audit artifacts and their recorded hashes.
  The original amendment hash is recorded in the v1 extraction summaries.

## Inference amendment 1.0.0 — 2026-11-11 (development, superseded)

- Added `inference_v1.json` without modifying frozen protocol `v1.json`.
- Made the grid-to-pixel variance conversion, inference clipping, input-space
  normalization, inclusive score boundary, and deterministic exact-anchor rule
  explicit.
- This amendment remains in development until the full paired validation audit
  passes; it must not yet be described as frozen.

## 1.0.0 — 2026-10-29

- Transcribed the experimental design from `thesis/sections/3-methodology.tex`.
- Bound the split definition to the downloaded official 160/67/91 directory
  memberships using canonical SHA-256 hashes.
- Froze candidate selection, shared features, missing-data encoding, spatial
  box selection, model grids, training limits, corruptions, seeds, evaluation
  constants, ablations, and statistical comparison settings.
- The detector checkpoint identities and hashes remain a Phase 2 artifact and
  are deliberately not part of this protocol file.
