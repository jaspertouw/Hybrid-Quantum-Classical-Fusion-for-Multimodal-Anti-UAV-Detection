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

## Original presence-fusion workflow

Verified development caches are under `outputs/caches/v1/clean/{train,val}` and
`outputs/caches/v1/pilot/val_3_per_sequence`. Frozen detector code and checkpoints
are unchanged. The shared feature transform fits the two spatial caps and eight
min/max ranges on clean training only; SVM standardization is fitted separately
inside each sequence-grouped calibration fold. Every model retains the fixed,
uncapped spatial box selector and learns presence/acceptance only.

```bash
python3 -m venv .venv/fusion
.venv/fusion/bin/python -m pip install -r environments/fusion_requirements.txt
.venv/fusion/bin/python scripts/successor/run_fusion_pilot.py
.venv/fusion/bin/python scripts/successor/select_fusion_models.py
```

Both commands refuse to overwrite a run and reject test inputs. The first
verifies references and fixed classical configurations on full clean validation
and the corruption pilot, with a small MLP/VQC/gate runtime check. Its five-epoch
neural results are not final model comparisons. The second searches all nine SVM
and nine forest settings using clean-validation EDTC at 0.5, with Brier ties,
then executes the prescribed full-sample MLP/VQC cross-entropy-versus-squared-error
loss pilot. Its models use the same 20,000 original training pairs; no corruption
outcome selects a configuration or loss.

Completed development runs:

- `original-fusion-pilot-v1`: feature/evaluator/reference and runtime verification.
- `clean-fusion-selection-v1`: selected SVM `C=10, gamma=0.1`, forest depth 8 /
  minimum leaf 5, DBF exponent 1, and shared expected-squared-error evidential loss.
  Both pilot architectures improved Brier without an EDTC reduction exceeding 0.01.

Artifacts live in `outputs/successor/protocol-1.0.0/runs/<run-id>/`. Read
`metrics/report.md` for the runtime pilot and `metrics/selection.json` for the
clean-development selection. Outputs include feature transforms, model states,
exact sample IDs, calibration folds, clipping counts, per-condition and natural-tag
scores, class support, reliability bins, error ranking and output hashes.

The quantum model is a differentiable, exact four-qubit complex128 statevector
simulation implemented in Torch, with the specified rotations, ordered CNOTs and
classical evidential head. Tests cover an analytic single-qubit identity, GHZ/ring
basis behaviour, parameter/input finite differences and save/load equivalence.
MLP/VQC architecture searches, final multi-seed training, full corruption caches,
ablations and noisy simulation remain required before freezing final artifacts.
The test split remains locked.

## Verification before further selection

```bash
PYTHONPATH=. .venv/fusion/bin/python -m pytest \
  tests/test_shared_fusion.py tests/test_neural_fusion.py tests/test_verification.py -q -W error
.venv/fusion/bin/python scripts/successor/verify_fusion_pipeline.py
```

`outputs/verification/fusion-local-v1/report.json` records two fresh synthetic
cache-to-export runs for all 12 methods, identical predictions/training histories,
36 independent dense-matrix circuit checks, the original fixed-box acceptance
oracle, explicit corruption deltas and a training-only sample-support study.
The circuit maximum absolute error was 1.11e-15 against a 1e-12 tolerance.
Multi-seed/condition EDTC bootstrap pairing is tested separately. AP and other
nonlinear metrics must be recomputed from frame-level predictions when bootstrapped;
the EDTC-only helper must not be reused for those metrics.

The 20,000-pair sample contains 126 absent frames across all 15 background-bearing
training sequences; the original SVM calibration folds contain 56/15/55 absent
examples. A fixed training-only holdout compared 20,000, 40,000 and all fitting
frames with fixed reference SVM/forest configurations. Neither EDTC nor specificity
improved with larger samples; classwise calibration changed modestly. The sample
cap remains 20,000 provisionally. These correlated frames and the holdout's three
background-bearing sequences do not establish universal minority-class adequacy.

The simulator choice is **Torch exact complex128 statevector**, checked against
independent NumPy dense unitaries, analytic identities and finite differences.
PennyLane is not a runtime dependency. `environments/fusion_constraints.txt`
records the complete local package set; the detector environment stays separate.
The CI workflow runs core and fusion checks in separate CPU jobs; configuration
is present, but no GitHub CI execution is claimed until it actually runs.

`fusion_environment_smoke.job` installs an isolated pinned CPU environment and
runs synthetic-only verification on Snellius, with a hard ten-minute limit and
no real dataset access. Long-work policy: notify the user and wait for approval
before uploading/submitting a workload expected to exceed 30 minutes, including
the aggregate proposed workload rather than splitting it to bypass that limit.
Final model choices remain provisional until the remaining readiness gates close.

Snellius smoke job **29372795** passed in **2m39s** (hard ten-minute limit).
It used a separate CPU environment with NumPy 1.26.4, SciPy 1.13.1,
scikit-learn 1.5.2 and Torch 2.4.1+cpu. All 12 methods repeated successfully;
the independent quantum-reference error was 7.22e-16. Downloaded artifacts and
environment export are `outputs/verification/smoke_29372795/` and
`outputs/verification/environment_29372795.txt`; all output/source hashes match.
No real dataset was accessed by that job and no workload over 30 minutes was
uploaded/submitted. Production arrays, resource/dependency/restart checks,
architecture searches, final seeds and test readiness remain outstanding.

Development task infrastructure is now available through
`scripts/successor/fusion_jobs.py` and the bounded CPU launcher
`scripts/successor/fusion_tasks.job`. Plans bind source hashes, the frozen
protocol, input artifacts and configurations. Incomplete neural fits restart
from their seed; completed fits are reused only after artifact verification.
Aggregation refuses missing or corrupted tasks. Stale locks require explicit
recovery with evidence that the owner is inactive. This is deterministic restart,
not continuation of optimizer state.

Local interruption/retry/reuse/corruption checks pass, alongside all 31 fusion
tests. The real-development benchmark is
`outputs/verification/fusion-jobs-benchmark-local-v1/`: three epochs each for
MLP16 and VQC depths 1–4, using the fixed 20,000-pair sample and full clean
validation. These operational timings do not select models. The local-only
20-configuration search plan is `outputs/verification/fusion-grid-plan-local-v1/`.
Its runtime estimate is stored with the benchmark; cluster timing and actual
SLURM array/dependency/interruption checks remain pending. The short launcher
explicitly rejects production grid plans. No new cluster workload was uploaded
or submitted during this infrastructure preparation.

The subsequent bounded Snellius pilot has now passed. Benchmark preparation,
five indexed fits and dependent aggregation completed (`29375916`–`29375918`).
The synthetic array intentionally failed task 0 (`29375913_0`), causing its
`afterok` aggregation (`29375914`) to be cancelled. Retry array `29376569`
restarted task 0, reused task 1 without changing its completion hash, and
aggregation `29376570` completed. Both downloaded result sets pass artifact and
source-contract verification. Evidence, accounting and the updated estimate:
`outputs/verification/fusion-cluster-results-v1/`.

Measured VQC training seconds/epoch at depths 1–4 were 2.11, 3.86, 5.71 and
7.54. Three-epoch timings are extrapolations, not full-fit measurements. The
20-configuration, seed-11 clean-development search has an approximately
89-minute serial budget before a factor-two safety allowance: approximately
177 worker-minutes total, or roughly 90–100 minutes with two workers, excluding
queue delays. Early stopping may reduce this. Proposed workers: two CPUs,
6 GiB, no GPU, 30-minute limit; Rome bills at least 16 CPUs per job.
Preparation/aggregation have ten-minute limits. The local-only
`fusion_grid.job` and `submit_fusion_grid.sh` implement this dependency chain.
This production search has not been uploaded or submitted and awaits user
approval under the aggregate workload policy. It does not include final seeds,
ablations, noisy simulation or test evaluation.
