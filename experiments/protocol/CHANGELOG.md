# Experimental protocol changelog

Frozen protocol files are immutable. A scientific change requires a copied file
with a new semantic version and a dated entry here. Path-only deployment values,
such as dataset and output roots, do not change the protocol version.

## 1.0.0 — 2026-10-29

- Transcribed the experimental design from `thesis/sections/3-methodology.tex`.
- Bound the split definition to the downloaded official 160/67/91 directory
  memberships using canonical SHA-256 hashes.
- Froze candidate selection, shared features, missing-data encoding, spatial
  box selection, model grids, training limits, corruptions, seeds, evaluation
  constants, ablations, and statistical comparison settings.
- The detector checkpoint identities and hashes remain a Phase 2 artifact and
  are deliberately not part of this protocol file.
