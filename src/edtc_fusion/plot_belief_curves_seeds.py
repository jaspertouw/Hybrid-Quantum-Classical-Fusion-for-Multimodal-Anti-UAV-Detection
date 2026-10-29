"""
plot_belief_curves_seeds.py — seed-AGGREGATED reliability & sparsification figures
==================================================================================
Companion to ``plot_belief_curves.py``.  That script overlays one curve per
fusion method from a *single* run folder.  This one aggregates over the 3 seeds
(0, 1, 2) so the thesis main-text figures carry seed variance instead of an
arbitrarily-chosen seed:

    * Reliability diagrams — per-confidence-bin empirical accuracy as the
      mean over seeds with ±1 std error bars; ECE reported as mean±std.
      -> reliability_seeds_<split>.png

    * Sparsification-error curves — per-method mean curve (oracle IoU −
      tracker IoU) with a shaded ±1 std band over seeds; AUSE reported as
      mean±std in the legend.
        * belief  E[P(UAV)] = cp_pres / ious_belief -> sparsification_belief_seeds_<split>.png
        * -sigma2_best       = cs_pres / ious_pres  -> sparsification_sigma_seeds_<split>.png

Per-frame aggregation, run-type detection, the sparsification fraction grid
(linspace(0.1, 1.0, 19)) and the reliability bin edges (linspace(0, 1, 11)) are
all reused verbatim from plot_belief_curves / eval_metrics, so a single-seed run
of this script reproduces the original figures (minus the zero-width band).

Folder layout (seeds 0-2, sibling dirs under the edtc_fusion base):
    belief methods (vacuous miss-encoding — the corrected main results):
        <split>_results_detector_only_{dbf,abf,naive}_peak_vacuous_seed<seed>
    DMS (belief-free, sigma figure only — non-vacuous naming):
        <split>_results_detector_only_dms_seed<seed>

Usage (mirrors plot_belief_curves.py)
-------------------------------------
    python -m src.edtc_fusion.plot_belief_curves_seeds \\
        --pretrained /home/ssuttorp/pretrained_models \\
        --val_root   /home/ssuttorp/Anti-UAV-RGBT/val \\
        --test_root  /home/ssuttorp/Anti-UAV-RGBT/test \\
        --out_dir    /home/ssuttorp/pretrained_models/edtc_fusion/figures

Pure CPU, no model loads.  Uses the Agg backend so it runs headless on SLURM.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")          # headless / SLURM-safe — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from src.edtc_fusion.plot_belief_curves import (  # noqa: E402
    _aggregate_run,
    _reliability_bins,
)
from lib.utils.eval_metrics import (  # noqa: E402
    compute_ece,
    compute_sparsification,
)

# ---------------------------------------------------------------------------
# Run inventory — (label, folder-stem); folder = <split>_results_detector_only_<stem>_seed<seed>
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2]

# Each variant is (label, seedN_stem, seed0_stem):
#   seedN_stem  -> folder  {split}_results_detector_only_{stem}_seed{seed}   (seeds 1,2,…)
#   seed0_stem  -> folder  {split}_results_detector_only_{stem}              (no suffix)
# Seed 0 was the original pre-seeds run, so its folder carries no "_seed0".
# Set seed0_stem=None when the seedN pattern already covers seed 0 (the vacuous
# reruns write an explicit "_seed0").  A stem is either a single string or a
# {"val":..., "test":...} dict when the per-split naming differs.
#
# IR-only / RGB-only are single evidential detectors: they emit b_uav/u/sigma2
# with NO fusion and NO b_bg suppression gate, so the vacuous miss-encoding
# correction does not change their outputs — their non-vacuous folders ARE the
# canonical version and mix cleanly with the vacuous fusion rows.  (Note the
# val/test asymmetry: val seeds 1,2 have no "_peak", test does; both seed-0
# originals do.)
#
# DBF/ABF/Naive use the vacuous miss-encoding — the corrected main result
# (see DBF/naive collapse note); their reruns include seed 0 explicitly.
BELIEF_VARIANTS = [
    ("Det-only IR",    {"val": "ir",  "test": "ir_peak"},  "ir_peak"),
    ("Det-only RGB",   {"val": "rgb", "test": "rgb_peak"}, "rgb_peak"),
    ("Det-only DBF",   "dbf_peak_vacuous",   None),
    ("Det-only ABF",   "abf_peak_vacuous",   None),
    ("Det-only Naive", "naive_peak_vacuous", None),
    # Gated: learned MLP gate fusion (belief-bearing).  Only non-vacuous folders
    # exist.  Its belief signals (conf_prob, 1-u) ignore b_bg -> the reliability
    # and belief-sparsification curves are encoding-INVARIANT, so they mix
    # cleanly with the vacuous rows.  Only the sigma-sparsification panel is
    # encoding-dependent (AUSE_sigma uses the gated emitted boxes): caption that
    # figure to note Gated uses the legacy dogmatic miss-encoding.
    ("Det-only Gated", "gated_peak",         "gated_peak"),
]
# Belief-free → appears only in the sigma-sparsification figure (it still has a
# localisation variance).  Non-vacuous naming: DMS does no belief fusion.
SIGMA_ONLY = [
    ("Det-only DMS",   "dms", "dms"),
]

# NOTE on Gated (included above, in BELIEF_VARIANTS): only non-vacuous folders
# exist for it.  This is sound for the reliability and belief-sparsification
# figures (its belief signals ignore b_bg -> encoding-invariant) but means its
# sigma-sparsification curve uses the legacy dogmatic miss-encoding while the
# DBF/ABF/Naive sigma curves use vacuous.  Footnote the sigma figure accordingly.

# Fixed fraction grid — must match eval_metrics.compute_sparsification's default
# so per-seed curves are evaluated at identical retention fractions.
FRACTIONS = np.linspace(0.1, 1.0, 19)


def _std(stack):
    """Sample std (ddof=1) over seeds; 0 when a single seed is present."""
    return stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros(stack.shape[1])


def _std_scalar(vals):
    vals = np.asarray(vals, dtype=float)
    return float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


# ---------------------------------------------------------------------------
# Per-method seed collection
# ---------------------------------------------------------------------------
def _collect_seed_runs(split, variants, base, dataset_root,
                       iou_threshold, cls_threshold):
    """{label: [per-seed data dict, ...]} — missing/empty seed folders skipped."""
    sp = split.lower()

    def _resolve(stem):
        # stem may be a plain string or a {"val":..., "test":...} per-split dict.
        return stem[sp] if isinstance(stem, dict) else stem

    out = {}
    for label, seedN_stem, seed0_stem in variants:
        seed_datas = []
        found = []
        for seed in SEEDS:
            if seed == 0 and seed0_stem is not None:
                # Original pre-seeds run: folder carries no "_seed0" suffix.
                folder = f"{sp}_results_detector_only_{_resolve(seed0_stem)}"
            else:
                folder = f"{sp}_results_detector_only_{_resolve(seedN_stem)}_seed{seed}"
            data = _aggregate_run(os.path.join(base, folder), dataset_root,
                                  iou_threshold, cls_threshold)
            if data is None:
                continue
            seed_datas.append(data)
            found.append(seed)
        if seed_datas:
            out[label] = seed_datas
            print(f"  {label:<16s} seeds found = {found}")
        else:
            print(f"  skip {label:<16s} (no seed folders for '{_resolve(seedN_stem)}')")
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_reliability_agg(split_name, runs_seeds, out_path, n_bins=10):
    """Grid of reliability diagrams — mean accuracy per bin with ±1 std bars."""
    runs_seeds = {k: [d for d in v if len(d["correctness"]) > 0]
                  for k, v in runs_seeds.items()}
    runs_seeds = {k: v for k, v in runs_seeds.items() if v}
    if not runs_seeds:
        print(f"  no belief-bearing runs for {split_name}; skip reliability.")
        return

    n = len(runs_seeds)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows),
                             squeeze=False)
    width = 1.0 / n_bins

    for ax, (label, seed_datas) in zip(axes.flat, runs_seeds.items()):
        # Per-seed bin accuracies and ECEs.
        acc_stack, conf_stack, eces = [], [], []
        centres = None
        for data in seed_datas:
            c, conf_mean, acc_mean, _ = _reliability_bins(
                data["conf_prob"], data["correctness"], n_bins)
            centres = c
            acc_stack.append(acc_mean)     # NaN in empty bins
            conf_stack.append(conf_mean)
            eces.append(compute_ece(data["conf_prob"], data["correctness"], n_bins))
        acc_stack = np.vstack(acc_stack)   # (S, n_bins), NaN where bin empty
        conf_stack = np.vstack(conf_stack)

        # Mean / std over seeds, ignoring bins a seed never populated.
        with np.errstate(invalid="ignore"):
            acc_mean = np.nanmean(acc_stack, axis=0)
            acc_std = (np.nanstd(acc_stack, axis=0, ddof=1)
                       if acc_stack.shape[0] > 1 else np.zeros(n_bins))
            conf_mean = np.nanmean(conf_stack, axis=0)
        good = ~np.isnan(acc_mean)
        acc_std = np.where(np.isnan(acc_std), 0.0, acc_std)

        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, zorder=1)
        ax.bar(centres[good], acc_mean[good], width=width * 0.9,
               color="#4C72B0", edgecolor="black", linewidth=0.4,
               alpha=0.85, label="accuracy (mean)", zorder=2)
        ax.errorbar(centres[good], acc_mean[good], yerr=acc_std[good],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=2,
                    zorder=4, label="±1 std over seeds")
        # Calibration gap (mean confidence − mean accuracy) shaded in red.
        ax.bar(centres[good], conf_mean[good] - acc_mean[good], width=width * 0.9,
               bottom=acc_mean[good], color="#C44E52", alpha=0.35,
               edgecolor="none", label="gap", zorder=3)

        ece_m = float(np.mean(eces))
        ece_s = _std_scalar(eces)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}\nECE = {ece_m:.4f} ± {ece_s:.4f}  "
                     f"(n={len(seed_datas)})", fontsize=10)
        ax.set_xlabel("confidence  (E[P(UAV)])", fontsize=8)
        ax.set_ylabel("empirical accuracy", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Reliability diagrams — {split_name}  "
                 f"(mean ± std over seeds {SEEDS}; gap to diagonal = ECE)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_sparsification_agg(split_name, runs_seeds, out_path,
                            conf_key="cp_pres", iou_key="ious_belief",
                            signal_name="belief  E[P(UAV)]"):
    """Per-method mean sparsification-error curve with a ±1 std band over seeds."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    cmap = plt.get_cmap("tab10")

    plotted = 0
    for idx, (label, seed_datas) in enumerate(runs_seeds.items()):
        gaps, auses = [], []
        for data in seed_datas:
            if len(data[iou_key]) == 0 or len(data[conf_key]) == 0:
                continue
            sp = compute_sparsification(data[iou_key], data[conf_key], FRACTIONS)
            gaps.append(sp["oracle_iou"] - sp["tracker_iou"])   # sparsification error
            auses.append(sp["ause"])
        if not gaps:
            continue
        gaps = np.vstack(gaps)             # (S, M)
        mean = gaps.mean(axis=0)
        std = _std(gaps)
        ause_m, ause_s = float(np.mean(auses)), _std_scalar(auses)
        color = cmap(idx % 10)
        ax.plot(FRACTIONS, mean, marker="o", markersize=3, linewidth=1.5,
                color=color,
                label=f"{label}  (AUSE = {ause_m:.4f} ± {ause_s:.4f}, "
                      f"n={len(gaps)})")
        ax.fill_between(FRACTIONS, mean - std, mean + std, color=color, alpha=0.18)
        plotted += 1

    if plotted == 0:
        print(f"  no usable runs for {split_name} sigma/belief figure; skipped.")
        plt.close(fig)
        return

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1,
               label="oracle (AUSE = 0)")
    ax.set_xlabel(f"retention fraction  (high-{signal_name} frames kept)")
    ax.set_ylabel("sparsification error  (oracle IoU − tracker IoU)")
    ax.set_title(f"Sparsification error — {split_name}  ·  signal: {signal_name}\n"
                 f"mean ± std over seeds {SEEDS}  ·  area = AUSE (lower is better)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_split(split_name, base, dataset_root, out_dir,
                 iou_threshold, cls_threshold):
    if dataset_root is None:
        return
    print(f"\n[{split_name}] collecting per-seed arrays …")
    belief_seeds = _collect_seed_runs(split_name, BELIEF_VARIANTS, base,
                                      dataset_root, iou_threshold, cls_threshold)
    sigma_extra = _collect_seed_runs(split_name, SIGMA_ONLY, base,
                                     dataset_root, iou_threshold, cls_threshold)
    if not belief_seeds and not sigma_extra:
        print(f"  no usable runs for {split_name}; nothing to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)
    tag = split_name.lower()

    plot_reliability_agg(
        split_name, belief_seeds,
        os.path.join(out_dir, f"reliability_seeds_{tag}.png"))
    plot_sparsification_agg(
        split_name, belief_seeds,
        os.path.join(out_dir, f"sparsification_belief_seeds_{tag}.png"),
        conf_key="cp_pres", iou_key="ious_belief",
        signal_name="belief  E[P(UAV)]")
    # sigma figure includes DMS (belief-free) alongside the belief methods.
    plot_sparsification_agg(
        split_name, {**belief_seeds, **sigma_extra},
        os.path.join(out_dir, f"sparsification_sigma_seeds_{tag}.png"),
        conf_key="cs_pres", iou_key="ious_pres",
        signal_name="-σ²  (localisation)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Seed-aggregated reliability & sparsification figures "
                    "(mean ± std over seeds 0-2).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pretrained",
                   help="Path to .../pretrained_models (we append /edtc_fusion).")
    g.add_argument("--base",
                   help="Direct path to the edtc_fusion results root.")
    p.add_argument("--val_root",  help="Anti-UAV-RGBT val split root.")
    p.add_argument("--test_root", help="Anti-UAV-RGBT test split root.")
    p.add_argument("--out_dir", default=None,
                   help="Where to write figures (default: <base>/figures).")
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--cls_threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    base = args.base or os.path.join(args.pretrained, "edtc_fusion")
    if not os.path.isdir(base):
        raise SystemExit(f"base dir not found: {base}")
    if args.val_root is None and args.test_root is None:
        raise SystemExit("provide at least one of --val_root / --test_root")
    out_dir = args.out_dir or os.path.join(base, "figures")

    _build_split("VAL",  base, args.val_root,  out_dir,
                 args.iou_threshold, args.cls_threshold)
    _build_split("TEST", base, args.test_root, out_dir,
                 args.iou_threshold, args.cls_threshold)
    print(f"\nFigures -> {out_dir}")


if __name__ == "__main__":
    main()
