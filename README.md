# Quantum and Classical Decision-Level Fusion for Multimodal Anti-UAV Detection

MSc thesis research building on the multimodal evidential-fusion work of
Suttorp et al. The research studies whether variational quantum representations improve
RGB--thermal decision-level fusion under clean, degraded, and missing-sensor conditions.

This repository is a successor to Sharanda Suttorp's
[`mm-uav-threat-detection`](https://github.com/sharandaa/mm-uav-threat-detection) repository. A
substantial part of the detector, tracker, fusion, training, and evaluation code is reproduced
from that project and retained here to provide the experimental baseline. New contributions for
this thesis will be kept clearly distinguishable from the inherited implementation.

> **Project status:** research design and repository setup. The inherited detector and fusion
> code is retained as a baseline; the new VQC, classical comparison models, degradation pipeline,
> and evaluation workflow have not yet been implemented. Do not interpret the baseline results
> below as results from this thesis.

## Current thesis scope

The project keeps Suttorp et al.'s RGB and thermal detectors fixed and operates on their
decision-level outputs. The intended learned fusion inputs are detector beliefs and uncertainty
features rather than raw images. Planned comparisons include a variational quantum classifier,
an evidential MLP, an RBF-SVM, a random forest, and the inherited naive/ABF/DBF/gated fusion
baselines. The thesis manuscript and its experiment-design chapter are maintained privately and
are not published in this repository.

Repository-specific setup and publication guidance is in [`docs/PROJECT_SETUP.md`](docs/PROJECT_SETUP.md).
The Anti-UAV dataset, pretrained checkpoints, generated outputs, and cluster logs are deliberately
excluded from version control.

---

## Inherited Suttorp baseline

The remainder of this document describes the repository as received from Suttorp et al. It is
kept for reproducibility and will be separated more fully from the successor workflow as the new
experiments are implemented.

### Original project title

**Uncertainty-Aware Multimodal Anti-UAV Detection via Evidential Fusion and Conflict-Discounted
Belief Aggregation**

MSc thesis code — **Sharanda Suttorp** (15841545), Master Information Studies (Data Science),
Faculty of Science, University of Amsterdam. Submitted 26 June 2026.
Supervisor: Dr. Seyed Sahand Mohammadi Ziabari.

---

## What this is

Anti-UAV perception systems must stay reliable when sensor streams degrade under occlusion,
fast motion, or modality-specific failure. Existing multimodal anti-UAV systems fuse RGB and
thermal streams *deterministically* — they cannot express doubt when the streams disagree.

This project extends **EDTC** (Evidential Detection and Tracking Collaboration) from
single-modality thermal-infrared to dual-stream **RGB + Thermal** perception, using:

- **Discounted Belief Fusion (DBF)** — converts inter-modal *conflict* into uncertainty mass
  before aggregating the per-stream class opinions in Subjective Logic space (semantic fusion).
- **Dynamic Modality Selection (DMS)** — resolves the bounding box by picking the modality with
  the lower normalized spatial variance, since RGB (1920×1080) and thermal (640×512) frames are
  spatially unaligned and direct box fusion is ill-posed (spatial fusion).
- **Per-modality evidential heads** (Dirichlet classification) and **Gaussian bounding-box heads**
  on top of independent per-stream YOLOv5 detectors.

The system commits to a detection only when the fused belief favours the UAV class **and** the
fused uncertainty falls below a calibrated gate threshold; otherwise it reports the target absent.

### Headline finding (the short version)

On the Anti-UAV benchmark, multimodal fusion consistently improves detection over either single
stream (test Acc **0.670** vs. 0.604 IR / 0.598 RGB) at real-time speed (≥38 FPS). However, **DBF
is empirically indistinguishable from undiscounted averaging (ABF) and naive late fusion** on this
benchmark — inter-modal conflict never exceeds `C ≤ 1.2×10⁻⁷`, leaving DBF's discounting step
inert. This null result is *structural*: the benchmark is presence-dominated and non-detecting
streams are encoded as vacuous, which jointly suppress conflict. The fused uncertainty is
well-calibrated (ECE 0.057) but is a weaker localization-failure detector than spatial variance
(AUROC 0.626 vs. 0.739). See the thesis for the full account.

> The full thesis PDF is the authoritative reference for methodology, results, and figures.

---

## Repository structure

```
mm-uav-threat-detection
├── src/
│   └── edtc_fusion/              # the fusion pipeline (main contribution)
│       ├── run_detector_only.py  # detector-only baselines + fusion (main experiments)
│       ├── run_edtc_fusion.py     # full detect+track multimodal variant (Appendix C)
│       ├── run_edtc_single.py     # single-modality detect+track
│       ├── run_naive_fusion.py    # naive late-fusion baseline
│       ├── run_gated_fusion.py    # learned-gate fusion baseline
│       ├── tracker*.py            # SiamCAR-based tracking branch variants
│       ├── train_gate.py          # trains the gated-fusion MLP
│       ├── rescore_belief_metrics.py  # ECE / AUROC / presence-discrimination metrics
│       ├── attribute_eval_all.py  # per-attribute robustness (TC/FM/SV/LI/LR/OV/OC)
│       ├── plot_belief_curves*.py # sparsification curves & reliability diagrams
│       └── print_metrics_tables.py
├── scripts/                      # SLURM (.job) + Python drivers for every experiment stage
│   ├── preprocess_antiuav*.py    # extract paired IR/RGB frames per sequence
│   ├── prepare_yolo_labels*.py   # build YOLO-format labels (IR + RGB)
│   ├── stage1_gaussian_*.job     # train Gaussian bbox heads
│   ├── stage2_evidential_*.job   # train evidential classification heads
│   ├── eval_detector_only_*.job  # run/eval each fusion variant (+ multi-seed)
│   ├── calibrate_detonly_*.job   # offline θ_cls threshold sweep on validation
│   └── analyse_*.py / aggregate_*.py  # conflict, separability, seed aggregation analyses
├── resources/
│   ├── EDTC/                     # vendored EDTC baseline (YOLOv5 + tracker) — see docs/MODIFICATIONS.md
│   └── SiamCAR/                  # vendored SiamCAR tracker
├── notebooks/                    # exploratory data analysis (Anti-UAV EDA, spatial demo)
├── logs/                         # SLURM job stdout from Snellius runs
├── environment.yml               # conda environment (thesis2026)
└── requirements.txt
```

---

## Data

Experiments use the **Anti-UAV** dataset: 318 paired RGB-Thermal sequences with ~580k annotated
bounding boxes, following the official splits — **train 160 / val 67 / test 91**. Each frame
carries a bounding box, a visibility flag, and seven challenge-attribute labels (thermal crossover,
fast motion, scale variation, low illumination, low resolution, out-of-view, occlusion).

The dataset is **not** included in this repository. Obtain Anti-UAV from its official source, then
extract paired frames into per-sequence subdirectories with `scripts/preprocess_antiuav.py`
(producing `ir_frames/`, `rgb_frames/`, `IR_label.json`, `visible_label.json`).

---

## Installation

```bash
# 1. Clone (the EDTC / SiamCAR baselines are vendored under resources/)
git clone https://github.com/sharandaa/mm-uav-threat-detection
cd mm-uav-threat-detection

# 2. Create the conda environment (PyTorch 2.1 / CUDA 12.1)
conda env create -f environment.yml
conda activate thesis2026

# (alternative, pip only)
pip install -r requirements.txt
```

The PreciseRoIPooling CUDA op used by the EDTC tracker is compiled on first use and requires
`ninja` plus a working CUDA toolchain (included in `environment.yml`).

---

## Usage

The experiments were run on **SURF Snellius** (NVIDIA A100, SLURM). Each stage has a `.job` script
under `scripts/` that wraps the corresponding Python entry point; the pipeline runs end-to-end as:

**1 — Preprocess & build labels**
```bash
python scripts/preprocess_antiuav.py          # extract paired IR/RGB frames
python scripts/prepare_yolo_labels.py         # IR YOLO labels
python scripts/prepare_yolo_labels_rgb.py     # RGB YOLO labels
```

**2 — Train per-modality detectors** (YOLOv5 baseline → Gaussian heads → evidential heads)
```bash
sbatch scripts/train_yolo_clean.job           # IR detector   (and *_rgb.job for RGB)
sbatch scripts/stage1_gaussian_ir.job         # Stage 1: Gaussian bbox heads
sbatch scripts/stage2_evidential_ir.job       # Stage 2: evidential classification heads
```

**3 — Run the fusion experiments** (detector-only configuration — the main results)
```bash
# single-modality and fused variants: ir | rgb | dbf | abf | naive | gated | dms
python -m src.edtc_fusion.run_detector_only --mode dbf \
    --dataset_root /path/to/Anti-UAV-RGBT/val \
    --results_dir  /path/to/results/val_dbf \
    --yolo_ir      /path/to/evidential_ir_stage2_best.pt \
    --yolo_rgb     /path/to/evidential_rgb_stage2_best.pt
```

**4 — Calibrate, score, and aggregate**
```bash
python scripts/calibrate_detonly_threshold.py   # offline θ_cls sweep on validation
python -m src.edtc_fusion.rescore_belief_metrics # ECE / AUROC / presence discrimination
python -m src.edtc_fusion.attribute_eval_all     # per-attribute robustness
python -m src.edtc_fusion.print_metrics_tables   # reproduce the results tables
```

Each `run_detector_only` call logs, per frame, the DMS box, the selected modality, the per-class
beliefs and `u_fused` *before* the gate is applied, so thresholds can be swept offline against the
logged values without re-running inference. Multi-seed variants (`*_seeds.job`) reproduce the
mean ± std reported in the thesis (seeds 0–2).

### Fusion modes

| `--mode` | Description |
|----------|-------------|
| `ir`    | Single-modality thermal YOLO |
| `rgb`   | Single-modality visible YOLO |
| `dbf`   | Dual YOLO + belief fusion + suppression gate + DMS |
| `naive` | Dual YOLO + averaged beliefs |
| `gated` | Dual YOLO + learned MLP gate fusion |
| `dms`   | Dual YOLO + DMS box selection only (no semantic fusion) |

Average Belief Fusion (ABF, no conflict discounting) is run as `--mode dbf --fusion abf`; the DBF
conflict-discount exponent is set with `--lam` (λ).

---

## How it works

The detection branch runs two independent YOLOv5 detectors (no shared parameters, so a degraded
stream cannot corrupt the other's representation). Each stream is fitted with:

- an **evidential head** (linear + ReLU) producing `K=2` non-negative evidence values, interpreted
  as Dirichlet concentration parameters → a Subjective Logic opinion `(b_UAV, b_bg, u)`;
- a **Gaussian head** predicting mean and log-variance per box coordinate, trained with a Gaussian
  negative log-likelihood loss (variance-calibrated localization).

**Semantic fusion (DBF).** The two stream opinions are dependent (same scene), so Dempster's rule
is inappropriate. DBF first computes inter-modal conflict `C`, derives a discount factor
`η = (1 − Cᵏ)^(1/k)`, rescales each stream's beliefs (absorbing the residual into uncertainty),
then aggregates via Average Belief Fusion. When streams conflict, `η → 0` drives `u_fused → 1` —
the system defaults to maximal uncertainty rather than a confidently wrong prediction.

**Spatial fusion (DMS).** Because RGB and thermal frames are unaligned and differ in resolution,
the box from the modality with the lower normalized spatial variance is selected each frame.

**Gated output.** A box is emitted iff `b_fused(UAV) > b_fused(bg)` **and** `u_fused < θ_cls`;
otherwise the target is reported absent. `θ_cls` is calibrated by an offline sweep on validation.

Training is two-stage: the Gaussian heads are trained first on the frozen YOLOv5 backbone, then the
evidential head is trained with the rest of the network frozen (KL regularization annealed linearly
0→1 over the first 10 epochs).

A full **detect+track** variant — adding a SiamCAR-based local tracking branch that carries the same
evidential and Gaussian heads — is implemented under `src/edtc_fusion/run_edtc_fusion.py` and
documented in Appendix C of the thesis. The detector-only configuration matched or exceeded it in
preliminary validation and was adopted for the main results.

---

## Acknowledgements

This work builds on several open-source projects:

- **EDTC** — Evidential Detection and Tracking Collaboration (the single-modality thermal baseline
  this thesis extends). Vendored under [`resources/EDTC/`](resources/EDTC/).
- **YOLOv5** by Ultralytics — the detection backbone. <https://github.com/ultralytics/yolov5>
- **SiamCAR** — the Siamese tracking branch used in the detect+track variant.
  Vendored under [`resources/SiamCAR/`](resources/SiamCAR/).
- **PreciseRoIPooling** — CUDA RoI-pooling op used by the EDTC tracker.

The methodology draws on Evidential Deep Learning (Sensoy et al.), Subjective Logic (Jøsang),
Discounted Belief Fusion (Bezirganyan et al.), and the Gaussian bounding-box formulation
(He et al.). Full citations are in the thesis references.

Computing infrastructure was provided by **SURF Snellius** (Dutch national supercomputer).
