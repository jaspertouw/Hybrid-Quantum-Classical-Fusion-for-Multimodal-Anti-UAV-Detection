# Experimental protocol changelog

Frozen protocol files are immutable. A scientific change requires a copied file
with a new semantic version and a dated entry here. Path-only deployment values,
such as dataset and output roots, do not change the protocol version.

## Frame-presence opinions 2.0.0-development

- `opinion_calibration_v1.json` declares a CPU-only evidence-scaling calibration
  check with disjoint head-fit/calibration/evaluation training sequences. Preserve
  valid Dirichlet opinions, fit weighted NLL with an identity penalty, and require
  both-class recognition together with probability improvements. Report all saved
  heads/seeds; calibrated probabilities do not certify evidential uncertainty.
- `opinion_objectives_v1.json` fixes a CPU-only natural-prior versus balanced
  class-weight comparison after the expanded diagnostic failed. Split original
  training sequences once, standardize fitting frames only, report all three
  seeds and raw held-out training probability quality. No official validation
  features, corruption/conflict objective or best-seed selection enters this step.
- Follow-up `presence_diagnostic_v1.json` predeclares a clean-only 128 versus
  2048-per-class nested training comparison after pilot `29131516` failed its
  thermal learning gate. Architecture, objective and validation keys stay fixed;
  save pooled features and report training/validation class and sequence support.
- Add `presence_opinions_v1.json` as a development specification, leaving all
  frozen v1 detector, inference and corruption artifacts unchanged.
- Pilot sensor-specific UAV/background supervision on pooled frozen neck
  features; retain semantic opinions without a candidate box. Sensor dropout
  remains vacuous. Existing candidate selection/boxes remain the reference.
- Preserve the inherited conflict operator. Require separate both-class and
  opposite-decision diagnostics; positive conflict alone is not sufficient.
- No production cache migration or final training schedule is approved by a
  successful pilot execution. Review scientific gates before freezing v2.

## Corruption specification 1.0.0 — 2026-11-11 (frozen)

- Implement all 30 predeclared image corruptions plus clean and two sensor dropouts.
- Specify float32 resized-content operations, unchanged letterbox padding,
  reflection without repeated edge pixels, and keyed SHA-256/PCG64 noise.
- Preserve the Phase 2 sources and completion record. Corruptions use a separate
  adapter and do not round pixels back to uint8 before inference.
- Job `29129768` passed three validation pairs across all conditions, including
  repeatability, unchanged-sensor and dropout checks. All three visual panels
  were inspected. Hashes and exact validation scope are in `corruptions_v1.json`.
- Full cache/pilot runs remain later work and were not submitted for this freeze.

## Inference amendment 1.1.0 — 2026-11-11 (frozen settings)

- Specify per-image auto-letterboxing directly from decoded frames, independent
  of batch composition; corrected extraction goes to `validation_candidates/v2`.
- Define frame-weighted EDTC as the frame-count-weighted mean of sequence
  scores. Preserve earlier pooled-component outputs as diagnostics.
- Preserve completed 1.0.0 extraction/audit artifacts and their recorded hashes.
  The original amendment hash is recorded in the v1 extraction summaries.
- Disable TF32 and use deterministic FP32 inference. Batch-128 parity job
  `29127443` passed both modalities over 201 predeclared frames each; tolerances
  were unchanged. Canonical full-split certification remains a separate Phase 2
  completion artifact, created only after extraction/audit success.

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
