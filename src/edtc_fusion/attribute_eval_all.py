"""
attribute_eval_all.py — Attribute robustness, every det-only variant at once
=============================================================================
Driver around rescore_attribute_metrics: scores ALL detector-only variants
(IR, RGB, DBF, ABF, Naive, Gated, DMS) across ALL available seeds, aggregates
mean +/- std per attribute, and prints ONE combined table per selected metric:

    rows    = attribute subsets (TC, FM, SV, LI, LR, OV, OC, TC tiers)
    columns = variants
    cell    = mean +/- std over seeds

Folder names follow print_metrics_tables.py's inventory (seed 0 = no-suffix
dir, seeds 1, 2 = *_seed{N}).  Missing dirs are skipped with a note, so the
script can run before every seed has finished.

Default metric selection (keep the thesis table small):

    acc    EDTC paper Acc            ece  calibration
    u      mean vacuity              rho  Spearman rho(u, IoU)
           (condition-awareness)          (uncertainty informativeness)

AUSE-belief is deliberately NOT a default — the main results section reports
AUSE-sigma only, so per-attribute AUSE-belief would introduce a metric used
nowhere else.  Add ``--metrics ... ause_sigma`` for per-attribute AUSE
consistency with the main tables (also gives DMS a real uncertainty value).

All metrics for all variants are always written to the combined JSON
(``attribute_eval_<split>.json`` in --base), so nothing is lost by being
selective in the printed tables — add e.g. ``--metrics acc miou ece auroc
ause u rho`` to print more.

Usage
-----
    python -m src.edtc_fusion.attribute_eval_all \\
        --base      /home/ssuttorp/pretrained_models/edtc_fusion \\
        --data_root /home/ssuttorp/Anti-UAV-RGBT \\
        --split     test            # or: val, both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EDTC_ROOT = _REPO_ROOT / "resources" / "EDTC"
for _p in (str(_REPO_ROOT), str(_EDTC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.edtc_fusion.print_metrics_tables import (  # noqa: E402
    VAL_RUNS,
    TEST_RUNS,
    _seed_runs,
)
from src.edtc_fusion.rescore_attribute_metrics import (  # noqa: E402
    ATTRIBUTE_ORDER,
    ATTRIBUTE_NAMES,
    _score_run,
    aggregate_seed_rows,
)

# Printed-table metric selection: short flag -> (agg key, table title).
METRIC_CHOICES = {
    "acc":        ("acc_edtc",          "Acc (EDTC paper formula, per-seq mean)"),
    "acc_thr":    ("accuracy_threshold", "Accuracy (IoU>=thr / bg-vote correctness)"),
    "miou":       ("mean_iou_present",  "Mean IoU (present frames)"),
    "ece":        ("ece_conf_prob",     "ECE (conf_prob)  [lower better]"),
    "auroc":      ("auroc_conf_prob",   "AUROC (conf_prob)"),
    "ause":       ("ause_conf_prob",    "AUSE (conf_prob)  [lower better]"),
    "ause_sigma": ("ause_conf_sigma",   "AUSE (sigma2)  [lower better]"),
    "u":          ("mean_u_all",        "Mean u_fused (all frames)"),
    "u_pres":     ("mean_u_present",    "Mean u_fused (present frames)"),
    "rho":        ("spearman_u_iou",    "Spearman rho(u, IoU)  [negative better]"),
}
DEFAULT_METRICS = ["acc", "ece", "u", "rho"]

VARIANT_LABELS = ["Det-only IR", "Det-only RGB", "Det-only DBF", "Det-only ABF",
                  "Det-only Naive", "Det-only Gated", "Det-only DMS"]
SHORT = {l: l.replace("Det-only ", "") for l in VARIANT_LABELS}


def _variant_dirs(split: str, seeds):
    """{variant short name: [folder name per seed]} from the shared inventory."""
    seed0 = {lbl: name for lbl, name in (VAL_RUNS if split == "val" else TEST_RUNS)}
    out = {}
    for lbl in VARIANT_LABELS:
        dirs = []
        for s in seeds:
            inv = seed0 if s == 0 else dict(_seed_runs(split, s))
            if lbl in inv:
                dirs.append(inv[lbl])
        out[SHORT[lbl]] = dirs
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Attribute robustness for all det-only variants x seeds.")
    p.add_argument("--base", required=True,
                   help="Dir holding the *_results_* run folders "
                        "(e.g. .../pretrained_models/edtc_fusion)")
    p.add_argument("--data_root", required=True,
                   help="Anti-UAV-RGBT root (contains val/, test/, label_new/)")
    p.add_argument("--split", default="test", choices=["val", "test", "both"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   choices=sorted(METRIC_CHOICES),
                   help=f"Metrics to print as combined tables "
                        f"(default: {' '.join(DEFAULT_METRICS)}); the JSON "
                        f"always contains every metric.")
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument("--min_seq", type=int, default=3)
    return p.parse_args()


def _fmt_cell(entry, key):
    m, s = entry.get(key + "_mean"), entry.get(key + "_std")
    if m is None:
        return "-"
    return f"{m:.4f}" if s is None else f"{m:.4f}+-{s:.4f}"


def _print_combined(split, metric_key, title, variant_agg, order, min_seq):
    variants = list(variant_agg)
    w_head, w_num = 9, 16
    print()
    print(f"== {split.upper()} — {title} ==")
    print("attr".rjust(w_head) + "".join(v.rjust(w_num) for v in variants))
    print("-" * (w_head + w_num * len(variants)))
    for name in order:
        cells = [name.rjust(w_head)]
        unstable = False
        for v in variants:
            entry = variant_agg[v].get(name)
            cells.append(("-" if entry is None
                          else _fmt_cell(entry, metric_key)).rjust(w_num))
            unstable = unstable or (entry or {}).get("unstable", False)
        print("".join(cells) + ("  *" if unstable else ""))
    print(f"* fewer than {min_seq} sequences — treat as unstable")


def _run_split(split: str, args):
    dataset_root = os.path.join(args.data_root, split)
    attr_path = os.path.join(args.data_root, "label_new", split + ".json")
    if not os.path.isfile(attr_path):
        raise SystemExit(f"attribute file not found: {attr_path}")
    with open(attr_path) as f:
        seq_attrs = json.load(f)

    score_args = SimpleNamespace(iou_threshold=args.iou_threshold,
                                 cls_threshold=args.cls_threshold,
                                 min_seq=args.min_seq)

    print(f"\n{'#' * 70}\n# {split.upper()} split — seeds {args.seeds}\n{'#' * 70}")

    variant_agg = {}        # variant -> {attr: agg entry}
    variant_meta = {}
    union_order = ["ALL"]
    for variant, dir_names in _variant_dirs(split, args.seeds).items():
        all_rows, used_dirs = [], []
        for name in dir_names:
            rd = os.path.join(args.base, name)
            if not os.path.isdir(rd):
                print(f"[skip] {name} (missing)")
                continue
            print(f"--- scoring {name}")
            _, rows, _ = _score_run(rd, dataset_root, seq_attrs,
                                    score_args, verbose=False)
            all_rows.append(rows)
            used_dirs.append(name)
        if not all_rows:
            print(f"[skip] variant {variant}: no result dirs found")
            continue
        order, agg = aggregate_seed_rows(all_rows)
        variant_agg[variant] = agg
        variant_meta[variant] = {"dirs": used_dirs, "n_seeds": len(used_dirs)}
        for a in order:
            if a not in union_order:
                union_order.append(a)

    if not variant_agg:
        raise SystemExit(f"No variant produced results for split '{split}'.")

    union_order = ["ALL"] + sorted([a for a in union_order if a != "ALL"],
                                   key=lambda a: (ATTRIBUTE_ORDER.index(a)
                                                  if a in ATTRIBUTE_ORDER else 99, a))

    out = {
        "split":           split,
        "base":            args.base,
        "seeds_requested": args.seeds,
        "iou_threshold":   args.iou_threshold,
        "cls_threshold":   args.cls_threshold,
        "min_seq":         args.min_seq,
        "note": ("Per variant: each seed's frames pooled per attribute, then "
                 "mean/std (sample, ddof=1) over seeds.  Attribute subsets "
                 "overlap (multi-label, sequence-level).  spearman_u_iou: "
                 "negative = high uncertainty tracks low IoU."),
        "attribute_names": ATTRIBUTE_NAMES,
        "variants":        {v: {**variant_meta[v], "per_attribute": variant_agg[v]}
                            for v in variant_agg},
    }
    out_path = os.path.join(args.base, f"attribute_eval_{split}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    n_seeds = {v: variant_meta[v]["n_seeds"] for v in variant_agg}
    print(f"\nSeeds found per variant: {n_seeds}")
    for short in args.metrics:
        key, title = METRIC_CHOICES[short]
        _print_combined(split, key, title, variant_agg, union_order,
                        args.min_seq)
    print(f"\nCombined summary -> {out_path}")


def main():
    args = parse_args()
    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split in splits:
        _run_split(split, args)


if __name__ == "__main__":
    main()
