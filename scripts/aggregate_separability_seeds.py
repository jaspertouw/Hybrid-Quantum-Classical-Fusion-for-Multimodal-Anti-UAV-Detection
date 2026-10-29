"""
aggregate_separability_seeds.py
================================
Aggregate the det-only uncertainty-separability results across seeds.

Reads the per-config separability_detonly.json files written by
analyse_uncertainty_separability_detonly.py for each seed:

    seed 0 : reports/separability_detonly/<cfg>_val/separability_detonly.json
    seed N : reports/separability_detonly_seedN/<cfg>_val/separability_detonly.json

and prints, per config and per signal (u_fused, b_uav, sigma2_best), the
failure-detection AUROC as mean +/- std (population std, ddof=0 — same
convention as the seed tables in print_metrics_tables.py), plus a LaTeX
table body matching the tracker separability table layout.

Usage (on Snellius, after the seed-0 and seeds-1/2 separability jobs):
    python scripts/aggregate_separability_seeds.py \
        --reports_root /home/ssuttorp/mm-uav-threat-detection/reports \
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONFIGS = ["ir", "rgb", "dbf", "abf", "naive", "gated"]
LABELS = {
    "ir": "Det-only IR", "rgb": "Det-only RGB", "dbf": "Det-only DBF",
    "abf": "Det-only ABF", "naive": "Det-only Naive", "gated": "Det-only Gated",
}
SIGNALS = ["u_fused", "b_uav", "sigma2_best"]


def seed_report_dir(reports_root: Path, seed: int) -> Path:
    suffix = "" if seed == 0 else f"_seed{seed}"
    return reports_root / f"separability_detonly{suffix}"


def load_seed_config(reports_root: Path, seed: int, cfg: str, split: str):
    p = seed_report_dir(reports_root, seed) / f"{cfg}_{split}" / "separability_detonly.json"
    if not p.is_file():
        print(f"[warn] missing: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def fmt(vals: list[float]) -> str:
    if not vals:
        return "--"
    a = np.asarray(vals, float)
    if a.size == 1:
        return f"{a[0]:.3f}"
    return f"{a.mean():.3f} +/- {a.std(ddof=0):.3f}"


def fmt_latex(vals: list[float]) -> str:
    if not vals:
        return "--"
    a = np.asarray(vals, float)
    if a.size == 1:
        return f"{a[0]:.3f}"
    return f"{a.mean():.3f} $\\pm$ {a.std(ddof=0):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_root", type=Path, default=Path("reports"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    args = ap.parse_args()

    # table[cfg][signal] -> list of per-seed AUROCs; counts for sanity
    table: dict[str, dict[str, list[float]]] = {
        c: {s: [] for s in SIGNALS} for c in args.configs}
    n_frames: dict[str, list[int]] = {c: [] for c in args.configs}
    used_seeds: dict[str, list[int]] = {c: [] for c in args.configs}

    for cfg in args.configs:
        for seed in args.seeds:
            data = load_seed_config(args.reports_root, seed, cfg, args.split)
            if data is None:
                continue
            used_seeds[cfg].append(seed)
            n_frames[cfg].append(data.get("n_frames", 0))
            for sig in SIGNALS:
                entry = data.get("signals", {}).get(sig)
                if entry is not None and entry.get("auroc_failure") is not None:
                    table[cfg][sig].append(float(entry["auroc_failure"]))

    # ---- console table -----------------------------------------------------
    w = 22
    print(f"\nDet-only separability AUROC (failure detection), "
          f"split={args.split}, seeds={args.seeds}")
    header = f"{'config':14s}" + "".join(f"{s:>{w}s}" for s in SIGNALS) + f"{'seeds':>12s}"
    print(header)
    print("-" * len(header))
    for cfg in args.configs:
        row = f"{LABELS.get(cfg, cfg):14s}"
        for sig in SIGNALS:
            row += f"{fmt(table[cfg][sig]):>{w}s}"
        row += f"{str(used_seeds[cfg]):>12s}"
        print(row)

    # ---- LaTeX body ----------------------------------------------------------
    print("\n% LaTeX table body (AUROC mean $\\pm$ std over seeds)")
    print("% config & $u_{fused}$ & $b_{uav}$ & $\\sigma^2_{best}$ \\\\")
    for cfg in args.configs:
        cells = " & ".join(fmt_latex(table[cfg][sig]) for sig in SIGNALS)
        print(f"{LABELS.get(cfg, cfg)} & {cells} \\\\")

    # ---- per-seed detail (for the appendix / sanity checking) ---------------
    print("\nPer-seed AUROC detail:")
    for cfg in args.configs:
        for sig in SIGNALS:
            vals = ", ".join(f"{v:.4f}" for v in table[cfg][sig])
            print(f"  {cfg:6s} {sig:12s} [{vals}]")
        if n_frames[cfg]:
            print(f"  {cfg:6s} n_frames per seed: {n_frames[cfg]}")


if __name__ == "__main__":
    main()
