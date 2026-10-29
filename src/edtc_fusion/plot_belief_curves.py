"""
plot_belief_curves.py — Reliability & sparsification figures for the belief metrics
===================================================================================
Companion to ``rescore_belief_metrics.py`` / ``print_metrics_tables.py``.

Where the table (``print_metrics_tables.py``) collapses each run to scalar
ECE / AUROC / AUSE, this script regenerates the *curves* behind those scalars and
overlays every run on a shared axis, one figure per split — the standard
calibration figures for a thesis:

    1. Reliability diagram  — per-confidence-bin empirical accuracy vs. the
       diagonal (perfect calibration).  The shaded gap to the diagonal is what
       ECE integrates.  One subplot per run (bar style); ECE annotated.

    2. Sparsification-error plots — for each run, the gap between the oracle
       curve (frames removed worst-IoU-first) and the tracker curve (removed
       least-confident-first), as a function of retention fraction.  The area
       under this gap is AUSE.  All runs overlaid; AUSE annotated in the legend.
       Two figures, one per confidence signal:
         * belief  E[P(UAV)] = b_uav + u/2   -> sparsification_belief_<split>.png
         * -sigma2_best       (localisation) -> sparsification_sigma_<split>.png

The per-frame arrays are rebuilt with the EXACT same logic the table uses
(``_per_sequence_arrays`` from ``rescore_belief_metrics``), so the curves are
guaranteed consistent with the reported scalars — no re-implementation.

Pure CPU, no model loads.  Uses the Agg backend so it runs headless on SLURM.

Usage
-----
    python -m src.edtc_fusion.plot_belief_curves \\
        --pretrained /home/ssuttorp/pretrained_models \\
        --val_root   /home/ssuttorp/Anti-UAV-RGBT/val \\
        --test_root  /home/ssuttorp/Anti-UAV-RGBT/test \\
        --out_dir    /home/ssuttorp/pretrained_models/edtc_fusion/figures

Plot only one split by omitting the other split's dataset root.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")          # headless / SLURM-safe — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from src.edtc_fusion.run_edtc_fusion import (  # noqa: E402
    discover_sequences,
    load_paired_sequence,
)
from src.edtc_fusion.rescore_belief_metrics import (  # noqa: E402
    _load_preds,
    _load_log,
    _detect_run_type,
    _per_sequence_arrays,
)
from src.edtc_fusion.print_metrics_tables import (  # noqa: E402
    VAL_RUNS,
    TEST_RUNS,
    _read_json,
)
from lib.utils.eval_metrics import (  # noqa: E402
    compute_ece,
    compute_sparsification,
)


# ---------------------------------------------------------------------------
# Per-run aggregation — mirrors rescore_belief_metrics.main()'s loop
# ---------------------------------------------------------------------------

def _aggregate_run(results_dir, dataset_root, iou_threshold, cls_threshold):
    """Rebuild the concatenated per-frame arrays for one finished run.

    Returns None if the run folder / logs are missing or hold no belief data.
    """
    summary = _read_json(os.path.join(results_dir, "summary.json"))
    if summary is None:
        return None
    run_type = _detect_run_type(summary)

    sequences = discover_sequences(dataset_root)
    # Two IoU arrays (see rescore_belief_metrics._per_sequence_arrays):
    #   ious_belief : present frames WITH beliefs   -> pairs with cp_pres (belief AUSE)
    #   ious_pres   : ALL present frames            -> pairs with cs_pres (sigma2 AUSE)
    # The belief-free DMS run has empty correctness/cp_pres but populated
    # ious_pres/cs_pres, so it appears only in the sigma2 sparsification figure.
    correct, cp_all = [], []
    ious_pres, ious_belief, cp_pres, cs_pres = [], [], [], []

    for seq in sequences:
        bbox_path = os.path.join(results_dir, seq + ".txt")
        log_path  = os.path.join(results_dir, seq + "_log.csv")
        seq_dir   = os.path.join(dataset_root, seq)
        if not (os.path.isfile(bbox_path) and os.path.isfile(log_path)):
            continue
        preds    = _load_preds(bbox_path)
        log_rows = _load_log(log_path)
        frames   = load_paired_sequence(seq_dir)
        if not frames:
            continue
        a = _per_sequence_arrays(preds, log_rows, frames, run_type,
                                 iou_threshold, cls_threshold)
        # Keep the sequence if it has EITHER belief data OR present-frame IoUs
        # (the latter alone suffices for the belief-free DMS run -> sigma2 AUSE).
        if len(a["correctness"]) == 0 and len(a["ious_present"]) == 0:
            continue
        correct.append(a["correctness"])
        cp_all.append(a["conf_prob"])
        ious_pres.append(a["ious_present"])     # all present frames
        ious_belief.append(a["ious_belief"])    # belief-bearing present frames
        cp_pres.append(a["cp_present"])
        cs_pres.append(a["cs_present"])         # -sigma2_best (localisation conf.)

    if not correct and not ious_pres:
        return None

    def _cat(chunks):
        return np.concatenate(chunks) if chunks else np.array([])

    return {
        "correctness": _cat(correct).astype(int) if correct else np.array([], dtype=int),
        "conf_prob":   _cat(cp_all),
        "ious_pres":   _cat(ious_pres),     # σ²-AUSE IoUs   (pairs with cs_pres)
        "ious_belief": _cat(ious_belief),   # belief-AUSE IoUs (pairs with cp_pres)
        "cp_pres":     _cat(cp_pres),
        "cs_pres":     _cat(cs_pres),
    }


def _reliability_bins(confs, correctness, n_bins=10):
    """Per-bin (centre, mean-confidence, empirical-accuracy, fraction-of-data).

    Binning matches compute_ece exactly (equal width, last bin right-inclusive)
    so the shaded gaps integrate to the reported ECE.
    """
    confs = np.asarray(confs, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    n = len(confs)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, conf_mean, acc_mean, weight = [], [], [], []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confs >= edges[i]) & (confs <= edges[i + 1])
        else:
            mask = (confs >= edges[i]) & (confs < edges[i + 1])
        centres.append(0.5 * (edges[i] + edges[i + 1]))
        if mask.sum() == 0:
            conf_mean.append(np.nan)
            acc_mean.append(np.nan)
            weight.append(0.0)
        else:
            conf_mean.append(float(confs[mask].mean()))
            acc_mean.append(float(correctness[mask].mean()))
            weight.append(float(mask.sum()) / n)
    return (np.array(centres), np.array(conf_mean),
            np.array(acc_mean), np.array(weight))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_reliability(split_name, runs_data, out_path, n_bins=10):
    """Grid of reliability diagrams — one subplot per run.

    Belief-free runs (e.g. DMS, with empty correctness) have no calibration to
    show and are skipped here; they still appear in the sigma2 sparsification.
    """
    runs_data = {k: v for k, v in runs_data.items() if len(v["correctness"]) > 0}
    if not runs_data:
        print(f"  no belief-bearing runs for {split_name}; skip reliability.")
        return
    n = len(runs_data)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows),
                             squeeze=False)
    width = 1.0 / n_bins

    for ax, (label, data) in zip(axes.flat, runs_data.items()):
        centres, conf_mean, acc_mean, weight = _reliability_bins(
            data["conf_prob"], data["correctness"], n_bins)
        ece = compute_ece(data["conf_prob"], data["correctness"], n_bins)

        # Perfect-calibration reference.
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, zorder=1)
        # Accuracy bars (only populated bins).
        good = weight > 0
        ax.bar(centres[good], acc_mean[good], width=width * 0.9,
               color="#4C72B0", edgecolor="black", linewidth=0.4,
               alpha=0.85, label="accuracy", zorder=2)
        # Calibration gap (accuracy minus confidence) shaded in red.
        ax.bar(centres[good], conf_mean[good] - acc_mean[good], width=width * 0.9,
               bottom=acc_mean[good], color="#C44E52", alpha=0.35,
               edgecolor="none", label="gap", zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}\nECE = {ece:.4f}", fontsize=10)
        ax.set_xlabel("confidence  (E[P(UAV)])", fontsize=8)
        ax.set_ylabel("empirical accuracy", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")

    # Hide any unused axes.
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Reliability diagrams — {split_name} "
                 f"(gap to diagonal = ECE)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_sparsification(split_name, runs_data, out_path,
                        conf_key="cp_pres", iou_key="ious_belief",
                        signal_name="belief  E[P(UAV)]"):
    """Overlaid sparsification-ERROR curves (oracle IoU − tracker IoU) per run.

    Confidence signal / paired IoU array (must match rescore_belief_metrics):
        belief : conf_key=cp_pres, iou_key=ious_belief  (present frames w/ beliefs)
        sigma2 : conf_key=cs_pres, iou_key=ious_pres    (ALL present frames)

    Lower curve = that confidence signal better predicts localization error.
    Area under each curve = AUSE (annotated in the legend).
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    cmap = plt.get_cmap("tab10")

    for idx, (label, data) in enumerate(runs_data.items()):
        if len(data[iou_key]) == 0 or len(data[conf_key]) == 0:
            continue
        sp = compute_sparsification(data[iou_key], data[conf_key])
        fractions = sp["fractions"]
        gap = sp["oracle_iou"] - sp["tracker_iou"]   # sparsification error
        ax.plot(fractions, gap, marker="o", markersize=3, linewidth=1.5,
                color=cmap(idx % 10),
                label=f"{label}  (AUSE = {sp['ause']:.4f})")

    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1,
               label="oracle (AUSE = 0)")
    ax.set_xlabel(f"retention fraction  (high-{signal_name} frames kept)")
    ax.set_ylabel("sparsification error  (oracle IoU − tracker IoU)")
    ax.set_title(f"Sparsification error — {split_name}  ·  signal: {signal_name}\n"
                 f"area under curve = AUSE  (lower is better)")
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

def _build_split(split_name, base, runs, dataset_root, out_dir,
                 iou_threshold, cls_threshold):
    if dataset_root is None:
        return
    print(f"\n[{split_name}] aggregating per-frame arrays …")
    runs_data = {}
    for label, folder in runs:
        rdir = os.path.join(base, folder)
        data = _aggregate_run(rdir, dataset_root, iou_threshold, cls_threshold)
        if data is None:
            print(f"  skip {label:<16s} (no data in {folder})")
            continue
        runs_data[label] = data
        print(f"  {label:<16s} {len(data['correctness'])} frames "
              f"({len(data['ious_pres'])} present)")
    if not runs_data:
        print(f"  no usable runs for {split_name}; nothing to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)
    tag = split_name.lower()

    # DMS is a pure box-selector: it performs no belief fusion, so any belief it
    # logs is a placeholder.  Its belief-based curve sits at ~0.0 and falsely
    # looks "perfect", so exclude it from the belief figures.  It IS meaningful
    # in the sigma2 figure (it still produces a localisation variance).
    belief_runs = {k: v for k, v in runs_data.items() if "DMS" not in k}

    plot_reliability(split_name, belief_runs,
                     os.path.join(out_dir, f"reliability_{tag}.png"))
    # Belief-based AUSE (conf_prob, DMS excluded) and sigma2-based AUSE
    # (-sigma2_best, DMS included) — one figure per confidence signal.
    plot_sparsification(split_name, belief_runs,
                        os.path.join(out_dir, f"sparsification_belief_{tag}.png"),
                        conf_key="cp_pres", iou_key="ious_belief",
                        signal_name="belief  E[P(UAV)]")
    plot_sparsification(split_name, runs_data,
                        os.path.join(out_dir, f"sparsification_sigma_{tag}.png"),
                        conf_key="cs_pres", iou_key="ious_pres",
                        signal_name="-σ²  (localisation)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Reliability & sparsification figures for the belief metrics.")
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

    # Detector-only runs only — skip the Det+Track rows for these figures.
    val_runs  = [(lbl, f) for lbl, f in VAL_RUNS  if lbl.startswith("Det-only")]
    test_runs = [(lbl, f) for lbl, f in TEST_RUNS if lbl.startswith("Det-only")]

    _build_split("VAL",  base, val_runs,  args.val_root,  out_dir,
                 args.iou_threshold, args.cls_threshold)
    _build_split("TEST", base, test_runs, args.test_root, out_dir,
                 args.iou_threshold, args.cls_threshold)
    print(f"\nFigures -> {out_dir}")


if __name__ == "__main__":
    main()
