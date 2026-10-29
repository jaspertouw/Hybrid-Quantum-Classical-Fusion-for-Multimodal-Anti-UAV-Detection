#!/usr/bin/env python
"""
aggregate_vacuous_belief_metrics.py — aggregating table for the vacuous rerun
=============================================================================
Reads the ``summary_belief_metrics.json`` files written by
``rescore_belief_metrics.py`` for the corrected missing-detection-encoding
runs (--miss_belief vacuous), and prints a mean±std table over seeds 0-2 for
the DBF and naive variants, per split.

Directory layout produced by rescore_belief_metrics_vacuous.job:
    <BASE>/<split>_results_detector_only_<variant>_peak_vacuous_seed<seed>/
        summary_belief_metrics.json

Usage
-----
    python scripts/aggregate_vacuous_belief_metrics.py \\
        --base /home/ssuttorp/pretrained_models/edtc_fusion \\
        --splits val test
"""

from __future__ import annotations

import argparse
import json
import math
import os

VARIANTS = ["abf"] #["naive", "dbf", "abf"]
SEEDS = [0, 1, 2]

# (json key, label, fmt) — order = row order in the table.
METRICS = [
    ("accuracy_threshold", "Accuracy (thresh.)", "{:.4f}"),
    ("mean_iou_present",   "Mean IoU (present)", "{:.4f}"),
    ("state_accuracy",     "State Accuracy",     "{:.4f}"),
    ("ap_50",              "AP@0.5",             "{:.4f}"),
    ("map_50_95",          "mAP@[0.5:0.95]",     "{:.4f}"),
    ("ece_conf_prob",      "ECE (conf_prob)",    "{:.4f}"),
    ("ece_conf_1mu",       "ECE (conf_1mu)",     "{:.4f}"),
    ("auroc_conf_prob",    "AUROC (conf_prob)",  "{:.4f}"),
    ("auroc_conf_1mu",     "AUROC (conf_1mu)",   "{:.4f}"),
    ("ause_conf_prob",     "AUSE (conf_prob)",   "{:.4f}"),
    ("ause_conf_1mu",      "AUSE (conf_1mu)",    "{:.4f}"),
    ("ause_conf_sigma",    "AUSE (sigma2)",      "{:.4f}"),
]

# Metrics living inside the "belief_classification" sub-dict.
CLS_METRICS = [
    ("accuracy",          "Cls accuracy",       "{:.4f}"),
    ("precision",         "Cls precision",      "{:.4f}"),
    ("recall",            "Cls recall",         "{:.4f}"),
    ("specificity",       "Cls specificity",    "{:.4f}"),
    ("false_alarm_rate",  "Cls false-alarm",    "{:.4f}"),
    ("f1",                "Cls F1",             "{:.4f}"),
    ("balanced_accuracy", "Cls balanced acc",   "{:.4f}"),
]


def _mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    if len(vals) > 1:
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        s = math.sqrt(var)
    else:
        s = 0.0
    return m, s, len(vals)


def _load(base, split, variant, seed):
    d = f"{split}_results_detector_only_{variant}_peak_vacuous_seed{seed}"
    path = os.path.join(base, d, "summary_belief_metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _collect(base, split, variant):
    """Return {json_key: [val_per_seed,...]} flattening belief_classification."""
    cols = {}
    found = []
    for seed in SEEDS:
        s = _load(base, split, variant, seed)
        if s is None:
            continue
        found.append(seed)
        for key, _, _ in METRICS:
            cols.setdefault(key, []).append(s.get(key))
        bc = s.get("belief_classification", {}) or {}
        for key, _, _ in CLS_METRICS:
            cols.setdefault("cls__" + key, []).append(bc.get(key))
    return cols, found


def _fmt_cell(m, s, n, fmt):
    if m is None:
        return f"{'n/a':>18}"
    body = f"{fmt.format(m)}±{fmt.format(s)}"
    if n < len(SEEDS):
        body += f"({n})"
    return f"{body:>18}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="dir holding the *_results_detector_only_* run folders "
                        "(e.g. .../pretrained_models/edtc_fusion)")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    args = p.parse_args()

    for split in args.splits:
        data = {v: _collect(args.base, split, v) for v in VARIANTS}

        print()
        print("=" * 78)
        print(f"VACUOUS belief-quality — split={split}   (mean±std over seeds "
              f"{SEEDS}; (n) flags <3 seeds)")
        for v in VARIANTS:
            print(f"   {v:<6}: seeds found = {data[v][1]}")
        print("=" * 78)
        header = f"{'Metric':<22}" + "".join(f"{v:>18}" for v in VARIANTS)
        print(header)
        print("-" * len(header))

        rows = [(k, lbl, fmt, False) for k, lbl, fmt in METRICS]
        rows += [("cls__" + k, lbl, fmt, True) for k, lbl, fmt in CLS_METRICS]

        for key, lbl, fmt, is_cls in rows:
            if is_cls and key == "cls__accuracy":
                print("-" * len(header))
            cells = []
            for v in VARIANTS:
                cols, _ = data[v]
                m, s, n = _mean_std(cols.get(key, []))
                cells.append(_fmt_cell(m, s, n, fmt))
            print(f"{lbl:<22}" + "".join(cells))
        print("=" * len(header))


if __name__ == "__main__":
    main()
