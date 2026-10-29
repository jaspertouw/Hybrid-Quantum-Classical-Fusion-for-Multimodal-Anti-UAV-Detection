"""
aggregate_dbf_lambda_seeds.py
=============================
Aggregate the DBF lambda sweep across seeds (0, 1, 2) on the detector-only
VAL runs: mean +- std per lambda for the belief-quality and operating-point
metrics.  CPU-only, no SLURM — run directly on the login node:

    python scripts/aggregate_dbf_lambda_seeds.py
    python scripts/aggregate_dbf_lambda_seeds.py --latex

Directory layout (VAL, peak aggregation):
    seed 0          : val_results_detector_only_dbf_peak_lam<tag>
    seeds 1/2       : val_results_detector_only_dbf_peak_seed<S>_lam<tag>
    lambda=1.0 s1/s2: val_results_detector_only_dbf_peak_seed<S>   (no suffix)

Each dir must contain summary_belief_metrics.json (rescore stage).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SEEDS = (0, 1, 2)
LAMS = ("0.25", "0.5", "1.0", "2.0", "4.0")

# (json path, short label)  — json path is dot-separated into the metrics file
METRICS = [
    ("state_accuracy",                          "Acc"),
    ("mean_iou_present",                        "mIoU"),
    ("ece_conf_prob",                           "ECE_p"),
    ("auroc_conf_prob",                         "AUROC_p"),
    ("ause_conf_prob",                          "AUSE_p"),
    ("belief_classification.precision",         "Prec"),
    ("belief_classification.recall",            "Rec"),
    ("belief_classification.specificity",       "Spec"),
    ("belief_classification.balanced_accuracy", "BalAcc"),
    ("belief_classification.false_alarm_rate",  "FAR"),
]


def run_dir(base: Path, seed: int, lam: str) -> Path:
    tag = lam.replace(".", "p")
    if seed == 0:
        return base / f"val_results_detector_only_dbf_peak_lam{tag}"
    if lam == "1.0":
        return base / f"val_results_detector_only_dbf_peak_seed{seed}"
    return base / f"val_results_detector_only_dbf_peak_seed{seed}_lam{tag}"


def get(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return float(d) if isinstance(d, (int, float)) else None


def mean_std(vals: list[float]) -> tuple[float, float]:
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)  # sample std, ddof=1
    return m, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=Path("/home/ssuttorp/pretrained_models/edtc_fusion"))
    ap.add_argument("--latex", action="store_true",
                    help="Also print LaTeX table rows (mean +- std).")
    args = ap.parse_args()

    # table[lam][label] = (mean, std, n_seeds)
    table: dict[str, dict[str, tuple[float, float, int]]] = {}
    for lam in LAMS:
        per_metric: dict[str, list[float]] = {lbl: [] for _, lbl in METRICS}
        for seed in SEEDS:
            f = run_dir(args.base, seed, lam) / "summary_belief_metrics.json"
            if not f.exists():
                print(f"[warn] missing {f}")
                continue
            data = json.loads(f.read_text())
            for path, lbl in METRICS:
                v = get(data, path)
                if v is not None:
                    per_metric[lbl].append(v)
        table[lam] = {lbl: (*mean_std(vs), len(vs)) if vs else (float("nan"), 0.0, 0)
                      for lbl, vs in per_metric.items()}

    labels = [lbl for _, lbl in METRICS]
    n_str = "/".join(str(table[lam][labels[0]][2]) for lam in LAMS)
    print(f"\nDBF lambda sweep — VAL, mean +- std over seeds (n per lambda: {n_str})\n")

    # Plain table
    print(f"{'lambda':>7}  " + "  ".join(f"{l:>16}" for l in labels))
    for lam in LAMS:
        cells = []
        for lbl in labels:
            m, s, n = table[lam][lbl]
            cells.append(f"{m:.4f}+-{s:.4f}" if n else "      —      ")
        print(f"{lam:>7}  " + "  ".join(f"{c:>16}" for c in cells))

    if args.latex:
        print("\n% LaTeX rows (lambda & " + " & ".join(labels) + " \\\\)")
        for lam in LAMS:
            cells = []
            for lbl in labels:
                m, s, n = table[lam][lbl]
                cells.append(f"${m:.4f} \\pm {s:.4f}$" if n else "---")
            row = f"{lam} & " + " & ".join(cells) + r" \\"
            print(r"\textbf{" + lam + "} & " + " & ".join(cells) + r" \\"
                  if lam == "1.0" else row)


if __name__ == "__main__":
    main()
