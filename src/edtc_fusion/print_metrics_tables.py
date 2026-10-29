"""
print_metrics_tables.py — Cross-run belief-metrics comparison
=============================================================
Aggregates ``summary_belief_metrics.json`` (and the original ``summary.json``
for paper-formula Acc) across every finished run and prints two side-by-side
tables — one for the val split, one for test — so the four baselines × two
architectures can be ranked at a glance.

Pure JSON reading — no GPU, no SLURM, no model loads.

Usage
-----
    python -m src.edtc_fusion.print_metrics_tables \\
        --pretrained /home/ssuttorp/pretrained_models

Or with a custom base dir:

    python -m src.edtc_fusion.print_metrics_tables \\
        --base /home/ssuttorp/pretrained_models/edtc_fusion

Pipe to a file for the thesis:

    python -m src.edtc_fusion.print_metrics_tables --pretrained ... > tables.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional


# ---------------------------------------------------------------------------
# Run inventory — folder names per split. Edit here if any layout changes.
# ---------------------------------------------------------------------------

VAL_RUNS = [
    ("Det-only IR",    "val_results_detector_only_ir_peak"),
    ("Det-only RGB",   "val_results_detector_only_rgb_peak"),
    ("Det-only DBF",   "val_results_detector_only_dbf_peak"),
    ("Det-only ABF",   "val_results_detector_only_abf_peak"),
    ("Det-only Naive", "val_results_detector_only_naive_peak"),
    ("Det-only Gated", "val_results_detector_only_gated_peak"),
    ("Det-only DMS",   "val_results_detector_only_dms"),
    ("Det+Track IR",   "val_results_single_ir_buav_logged"),
    ("Det+Track RGB",  "val_results_single_rgb_buav_logged"),
    ("Det+Track DBF",  "val_results_calibrated_cd0.550_ld0.035_ct0.529_lt0.564"),
    ("Det+Track ABF",  "val_results_abf_cd0.550_ld0.035_ct0.529_lt0.564"),
]

TEST_RUNS = [
    ("Det-only IR",    "test_results_detector_only_ir_peak"),
    ("Det-only RGB",   "test_results_detector_only_rgb_peak"),
    ("Det-only DBF",   "test_results_detector_only_dbf_peak"),
    ("Det-only ABF",   "test_results_detector_only_abf_peak"),
    ("Det-only Naive", "test_results_detector_only_naive_peak"),
    ("Det-only Gated", "test_results_detector_only_gated_peak"),
    ("Det-only DMS",   "test_results_detector_only_dms"),
    ("Det+Track IR",   "test_results_single_ir"),
    ("Det+Track RGB",  "test_results_single_rgb"),
    ("Det+Track DBF",  "test_results_dbf"),
    ("Det+Track ABF",  "test_results_abf"),
]


# Per-seed det-only folder names (seeds 1, 2 …).  Only the detector-only
# variants were re-run per seed; the Det+Track rows have no seed variants.
# NOTE the irregular naming inherited from the eval scripts:
#   val IR/RGB drop the "_peak" tag; test IR/RGB keep it; DMS has no "_peak".
def _seed_runs(split: str, seed: int):
    if split == "val":
        return [
            ("Det-only IR",    f"val_results_detector_only_ir_seed{seed}"),
            ("Det-only RGB",   f"val_results_detector_only_rgb_seed{seed}"),
            ("Det-only DBF",   f"val_results_detector_only_dbf_peak_seed{seed}"),
            ("Det-only ABF",   f"val_results_detector_only_abf_peak_seed{seed}"),
            ("Det-only Naive", f"val_results_detector_only_naive_peak_seed{seed}"),
            ("Det-only Gated", f"val_results_detector_only_gated_peak_seed{seed}"),
            ("Det-only DMS",   f"val_results_detector_only_dms_seed{seed}"),
        ]
    return [
        ("Det-only IR",    f"test_results_detector_only_ir_peak_seed{seed}"),
        ("Det-only RGB",   f"test_results_detector_only_rgb_peak_seed{seed}"),
        ("Det-only DBF",   f"test_results_detector_only_dbf_peak_seed{seed}"),
        ("Det-only ABF",   f"test_results_detector_only_abf_peak_seed{seed}"),
        ("Det-only Naive", f"test_results_detector_only_naive_peak_seed{seed}"),
        ("Det-only Gated", f"test_results_detector_only_gated_peak_seed{seed}"),
        ("Det-only DMS",   f"test_results_detector_only_dms_seed{seed}"),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _orig_acc(summary: Optional[dict]) -> Optional[float]:
    """Extract the paper-formula Acc the runner wrote. Field name varies by
    runner: detector-only/single use 'acc'; full fusion uses 'acc_dms'; a few
    older logs use 'acc_ir'."""
    if summary is None:
        return None
    return (summary.get("acc_dms")
            or summary.get("acc")
            or summary.get("acc_ir"))


def _load_row(base: str, label: str, folder: str) -> Optional[dict]:
    rdir = os.path.join(base, folder)
    bm   = _read_json(os.path.join(rdir, "summary_belief_metrics.json"))
    orig = _read_json(os.path.join(rdir, "summary.json"))
    if bm is None and orig is None:
        return None
    cls = (bm or {}).get("belief_classification", {})
    return {
        "name":        label,
        "folder":      folder,
        "acc_paper":   _orig_acc(orig),
        "state_acc":   (bm or {}).get("state_accuracy"),
        "mean_iou":    (bm or {}).get("mean_iou_present"),
        "ap50":        (bm or {}).get("ap_50"),
        "map":         (bm or {}).get("map_50_95"),
        "ece":         (bm or {}).get("ece_conf_prob"),
        "auroc":       (bm or {}).get("auroc_conf_prob"),
        "ause":        (bm or {}).get("ause_conf_prob"),
        "ause_sigma":  (bm or {}).get("ause_conf_sigma"),
        "cls_acc":     cls.get("accuracy"),
        "precision":   cls.get("precision"),
        "recall":      cls.get("recall"),
        "specificity": cls.get("specificity"),
        "f1":          cls.get("f1"),
        "bal_acc":     cls.get("balanced_accuracy"),
        "n_present":   cls.get("n_present_gt"),
        "n_absent":    cls.get("n_absent_gt"),
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

NAME_W = 16
NUM_W  = 8
HEADERS = ["Run", "Acc", "SA", "mIoU", "AP50", "mAP", "ECE", "AUROC", "AUSE",
           "AUSEsig", "ClsAcc", "Prec", "Recall", "Spec", "F1", "BalAcc"]
NUM_COLS = len(HEADERS) - 1
TOTAL_W = NAME_W + NUM_COLS * (NUM_W + 1)


def _cell(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    return f"{v:.4f}"


def _format_row(label: str, *vals) -> str:
    out = f"{label[:NAME_W]:<{NAME_W}}"
    for v in vals:
        out += f" {_cell(v):>{NUM_W}}"
    return out


def _format_header() -> str:
    out = f"{HEADERS[0]:<{NAME_W}}"
    for h in HEADERS[1:]:
        out += f" {h:>{NUM_W}}"
    return out


def print_table(title: str, base: str, runs):
    rows = [_load_row(base, lbl, folder) for lbl, folder in runs]
    present_rows = [r for r in rows if r is not None]
    if not present_rows:
        print(f"\n{title}: no data found under {base}\n")
        return

    bar = "=" * TOTAL_W
    print()
    print(bar)
    print(f" {title}")
    # Use the first row's GT counts for context (all rows in a split share them).
    np_, na = present_rows[0]["n_present"], present_rows[0]["n_absent"]
    if np_ is not None:
        print(f" GT frames: {np_} present, {na} absent  "
              f"(class imbalance ratio {np_ / max(na, 1):.1f}:1)")
    print(bar)
    print(_format_header())
    print("-" * TOTAL_W)
    for orig, row in zip(runs, rows):
        label = orig[0]
        if row is None:
            print(f"{label[:NAME_W]:<{NAME_W}}   (missing — folder not found)")
            continue
        print(_format_row(
            row["name"], row["acc_paper"], row["state_acc"], row["mean_iou"],
            row["ap50"], row["map"],
            row["ece"], row["auroc"], row["ause"],
            row["ause_sigma"],
            row["cls_acc"], row["precision"], row["recall"], row["specificity"],
            row["f1"], row["bal_acc"],
        ))
    print(bar)
    # Quick reading guide
    print(" Acc, SA, mIoU, AP50, mAP, AUROC, Prec, Recall, Spec, F1, BalAcc : higher is better")
    print(" ECE, AUSE, AUSEsig                                              : lower is better")
    print(" SA      = Anti-UAV State Accuracy (official benchmark metric)")
    print(" mIoU    = mean IoU over present frames")
    print(" AP50    = AP@0.5 ; mAP = mAP@[0.5:0.95] (emitted boxes @conf>=0.25, COCO 101-pt)")
    print(" AUSEsig = AUSE using -sigma2_best (localisation uncertainty)")
    print(bar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Cross-run belief-metrics comparison.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pretrained",
                   help="Path to .../pretrained_models (we append /edtc_fusion).")
    g.add_argument("--base",
                   help="Direct path to the edtc_fusion results root.")
    p.add_argument("--seed", type=int, nargs="*", default=None,
                   help="If given (e.g. --seed 1 2), print per-seed det-only "
                        "tables instead of the default seed-0 inventory.")
    return p.parse_args()


def main():
    args = parse_args()
    base = args.base or os.path.join(args.pretrained, "edtc_fusion")
    if not os.path.isdir(base):
        raise SystemExit(f"base dir not found: {base}")

    if args.seed:
        for s in args.seed:
            print_table(f"VAL split — det-only — seed {s}",  base, _seed_runs("val",  s))
            print_table(f"TEST split — det-only — seed {s}", base, _seed_runs("test", s))
    else:
        print_table("VAL split — belief-metrics comparison",  base, VAL_RUNS)
        print_table("TEST split — belief-metrics comparison", base, TEST_RUNS)


if __name__ == "__main__":
    main()
