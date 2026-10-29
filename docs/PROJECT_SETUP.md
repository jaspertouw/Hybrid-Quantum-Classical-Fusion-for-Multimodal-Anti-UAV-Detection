# Project setup and repository publication

This repository is a successor project to Suttorp et al.'s multimodal Anti-UAV thesis code,
published as [`sharandaa/mm-uav-threat-detection`](https://github.com/sharandaa/mm-uav-threat-detection).
A substantial part of this repository reproduces code from that project. The current research
keeps the inherited RGB and thermal detectors fixed and investigates quantum and classical
decision-level fusion under unequal sensor reliability.

## What is inherited and what is new

- `resources/`, most of `src/edtc_fusion/`, and the existing `scripts/` contain the inherited
  detector, tracker, fusion, training, and evaluation implementation.
- `thesis/` contains Jasper Touw's private thesis text and is intentionally excluded from Git.
- New experiment code should be added in clearly named modules rather than silently changing the
  inherited baselines. Every intentional baseline modification should be recorded in
  `docs/MODIFICATIONS.md` with its motivation and effect on reproducibility.
- The README's baseline results belong to Suttorp et al.; results for the successor thesis must be
  reported separately.

## Data and artifact layout

Large or restricted files must stay outside Git. A practical layout is:

```text
project repository/       source, configs, tests, thesis text
Anti-UAV-RGBT/            dataset (outside the repository)
pretrained_models/        detector checkpoints (ignored if placed in the repository)
outputs/                   cached predictions and evaluation output (ignored)
logs/                      local or SLURM logs (ignored)
```

Commands should receive dataset, checkpoint, and output locations as command-line arguments or
environment variables. Do not add a personal `/home/...` path to new code. Many inherited SLURM
scripts still contain `/home/ssuttorp/...`; treat those as provenance, not portable entry points.

## Environment

The inherited CUDA environment is defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate thesis2026
```

It targets Python 3.9, PyTorch 2.1, and CUDA 12.1 and is primarily intended for reproducing the
inherited detector pipeline on a Linux CUDA system. macOS cannot install the `pytorch-cuda`
package. The VQC and classical-fusion dependencies should be introduced in a separate environment
file when their implementation begins, so baseline reproducibility is not changed accidentally.

The top-level `requirements.txt` is an older CPU-oriented dependency snapshot and is not
equivalent to the Conda environment. Prefer `environment.yml` for inherited experiments.

The inherited GitHub Actions are marked manual-only. They target the old baseline dependency
stack and should not be treated as validation of the successor experiments. Replace them with a
small CPU-only test workflow when the new fusion modules are introduced.

## Checks before the first public push

1. Confirm that `git status --short --ignored` shows datasets, checkpoints, logs, outputs, local
   environments, and generated thesis PDFs as ignored.
2. Search staged files for credentials, personal absolute paths, and restricted data.
3. Inspect large staged files with `git ls-files -s`; GitHub rejects individual files over 100 MB.
4. Verify the reuse terms of every inherited component. `resources/EDTC/LICENSE` covers EDTC, but
   no repository-wide license was found, so do not add a new top-level license without resolving
   the rights to Suttorp's original repository and the vendored dependencies.
5. Credit Suttorp et al., EDTC, YOLOv5, SiamCAR, and PreciseRoIPooling in the README and thesis.

## Create and publish the Git repository

After choosing a GitHub repository name and visibility:

```bash
git init -b main
git add .
git status --short
git commit -m "Initialize thesis research repository"
git remote add origin git@github.com:<github-user>/<repository>.git
git push -u origin main
```

Create the empty GitHub repository without a generated README, `.gitignore`, or license so the
first push does not create unrelated history. SSH or HTTPS authentication must already be set up.
After pushing, compare the GitHub file list with the local staged-file review and ensure no data,
checkpoints, logs, or generated outputs appear.

## Deferred experiment setup

The following work is intentionally deferred until experiment implementation starts:

- a dedicated environment and lock/constraints file for PennyLane and the classical baselines;
- portable configuration for dataset/checkpoint/output paths;
- cached detector-output schema and provenance hashes;
- VQC, evidential MLP, SVM, forest, corruption, ablation, and evaluation modules;
- tests and CI that exercise the new fusion code without requiring the Anti-UAV dataset or a GPU.
