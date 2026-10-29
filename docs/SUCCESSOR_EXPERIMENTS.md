# Successor experiment structure

New thesis experiment code lives in `src/thesis_experiments/`. The inherited
baseline remains in `src/edtc_fusion/` and `resources/`; successor work must not
silently alter its behaviour.

The portable runtime configuration is `experiments/config/default.json`. It
contains only repository-relative defaults. Paths can be overridden without
editing tracked files:

```bash
python -m src.thesis_experiments.cli \
  --dataset-root /path/to/Anti-UAV-RGBT \
  --checkpoint-root /path/to/pretrained_models \
  --output-root /scratch/project/outputs \
  init --run-id local-structure-smoke \
  --input experiments/protocol/v1.json
```

Artifacts are grouped by protocol and run:

```text
outputs/successor/protocol-1.0.0/runs/<run-id>/
├── provenance.json
├── PARTIAL.json
├── manifests/
├── caches/
├── models/
├── metrics/
├── plots/
└── logs/
```

`provenance.json` records the Git commit and dirty state, protocol/configuration,
software/platform details, and hashes of declared input files. Initialization
always creates `PARTIAL.json`. A run becomes complete only through the explicit
`complete` command, which verifies and hashes every required non-empty output,
writes `COMPLETED.json` atomically, and removes `PARTIAL.json`.

```bash
python -m src.thesis_experiments.cli \
  complete --run-id local-structure-smoke \
  --required-output metrics/results.json
```

Completed run directories are immutable through this interface. Use a new run
ID rather than overwriting a completed result.
