"""
plot_edtc_threshold_distributions.py
====================================
Read the per-frame *_log.csv files emitted by run_edtc_fusion.py or
run_edtc_single.py and produce diagnostic plots for threshold calibration.

For each results directory:
  - Histogram of u and sigma2 split by state (DETECT vs TRACK).
  - Per-state percentile table (p50/p75/p90/p95/p99) printed to stdout and
    saved to threshold_stats.json.
  - Suggested threshold values: 95th percentile of the "good" operating range
    in each state.  Good = frames where the tracker did NOT switch.

Usage:
    python scripts/plot_edtc_threshold_distributions.py \
        --results_dir /path/to/val_results_dbf \
        [--results_dir /path/to/val_results_single_ir ...] \
        --out_dir /path/to/plots

Works for both fusion logs (columns: u_fused, sigma2_best) and single-modality
logs (columns: u, sigma2) — column names are auto-detected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


U_COLS     = ("u_fused", "u")
SIGMA_COLS = ("sigma2_best", "sigma2")


def detect_columns(header):
    u_col     = next((c for c in U_COLS     if c in header), None)
    sigma_col = next((c for c in SIGMA_COLS if c in header), None)
    if u_col is None or sigma_col is None:
        raise ValueError(f"Could not find u/sigma columns in header {header}")
    return u_col, sigma_col


def load_logs(results_dir: Path):
    """Walk *_log.csv files in results_dir; return (state, u, sigma2, switched) arrays."""
    states, us, sigmas, switched = [], [], [], []
    for csv_path in sorted(results_dir.glob("*_log.csv")):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            try:
                u_col, sigma_col = detect_columns(reader.fieldnames)
            except ValueError:
                continue
            for row in reader:
                u_raw = row.get(u_col, "")
                s_raw = row.get(sigma_col, "")
                if u_raw == "" or s_raw == "":
                    continue
                try:
                    u = float(u_raw)
                    s = float(s_raw)
                except ValueError:
                    continue
                if not (math.isfinite(u) and math.isfinite(s)):
                    continue
                states.append(row.get("state", ""))
                us.append(u)
                sigmas.append(s)
                sw_raw = row.get("switched", "")
                switched.append(int(sw_raw) if sw_raw not in ("", None) else 0)
    return (
        np.asarray(states),
        np.asarray(us, dtype=np.float64),
        np.asarray(sigmas, dtype=np.float64),
        np.asarray(switched, dtype=np.int32),
    )


def percentiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {p: float("nan") for p in (50, 75, 90, 95, 99)}
    return {p: float(np.percentile(values, p)) for p in (50, 75, 90, 95, 99)}


def plot_hist(ax, values: np.ndarray, label: str, color: str, bins=60, log_x=False):
    if values.size == 0:
        return
    if log_x:
        v = values[values > 0]
        if v.size == 0:
            return
        ax.hist(np.log10(v), bins=bins, alpha=0.55, label=f"{label} (n={values.size})",
                color=color)
    else:
        ax.hist(values, bins=bins, alpha=0.55, label=f"{label} (n={values.size})",
                color=color)


def process_one(results_dir: Path, out_dir: Path):
    print(f"\n=== {results_dir.name} ===")
    states, us, sigmas, switched = load_logs(results_dir)
    if states.size == 0:
        print(f"  (no usable log rows)")
        return None

    # Bucket: DETECT-non-switching, DETECT-switching (init), TRACK-non-switching, TRACK-switching (revert)
    is_detect = np.char.startswith(states.astype(str), "DETECT")
    is_track  = states == "TRACK"
    is_switch = switched == 1

    buckets = {
        "DETECT (stayed)":  (is_detect & ~is_switch),
        "DETECT->TRACK":    (is_detect &  is_switch),
        "TRACK  (stayed)":  (is_track  & ~is_switch),
        "TRACK ->DETECT":   (is_track  &  is_switch),
    }

    stats = {}
    for name, mask in buckets.items():
        stats[name] = {
            "n":      int(mask.sum()),
            "u":      percentiles(us[mask]),
            "sigma2": percentiles(sigmas[mask]),
        }
        if mask.sum() == 0:
            print(f"  {name:18s}  (empty)")
            continue
        print(f"  {name:18s}  n={mask.sum():6d}  "
              f"u  p50={stats[name]['u'][50]:.3f}  p95={stats[name]['u'][95]:.3f}  "
              f"sigma2 p50={stats[name]['sigma2'][50]:.3f}  p95={stats[name]['sigma2'][95]:.3f}")

    # Suggested thresholds: 95th percentile of the "good" non-switching frames
    # — i.e. the value you'd need to admit for the system to keep operating.
    # Detect thresholds are floored by the DETECT->TRACK frames (you must be at
    # least permissive enough to let those through), and track thresholds by
    # the TRACK (stayed) frames.
    suggested = {}
    if stats["DETECT->TRACK"]["n"] > 0:
        suggested["theta_cls_detect"] = stats["DETECT->TRACK"]["u"][95]
        suggested["theta_loc_detect"] = stats["DETECT->TRACK"]["sigma2"][95]
    if stats["TRACK  (stayed)"]["n"] > 0:
        suggested["theta_cls_track"] = stats["TRACK  (stayed)"]["u"][95]
        suggested["theta_loc_track"] = stats["TRACK  (stayed)"]["sigma2"][95]

    print("  suggested (p95 of good frames):")
    for k, v in suggested.items():
        print(f"    {k:18s} = {v:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{results_dir.name} — threshold diagnostics", fontsize=13)

    # Row 0: u histograms
    for col, mask_name in enumerate(("DETECT (stayed)", "TRACK  (stayed)")):
        ax = axes[0, col]
        for name, mask in buckets.items():
            if not name.startswith(mask_name.split()[0]):
                continue
            color = "C0" if "stayed" in name else "C3"
            plot_hist(ax, us[mask], name, color)
        ax.set_xlabel("u")
        ax.set_ylabel("count")
        ax.set_title(f"u  ({mask_name.split()[0]} state)")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    # Row 1: sigma2 histograms (log10 because tail is heavy)
    for col, mask_name in enumerate(("DETECT (stayed)", "TRACK  (stayed)")):
        ax = axes[1, col]
        for name, mask in buckets.items():
            if not name.startswith(mask_name.split()[0]):
                continue
            color = "C0" if "stayed" in name else "C3"
            plot_hist(ax, sigmas[mask], name, color, log_x=True)
        ax.set_xlabel("log10(sigma2)")
        ax.set_ylabel("count")
        ax.set_title(f"sigma2  ({mask_name.split()[0]} state)")
        ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = out_dir / f"{results_dir.name}_distributions.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"  -> {png_path}")

    json_path = out_dir / f"{results_dir.name}_stats.json"
    with open(json_path, "w") as f:
        json.dump({"stats": stats, "suggested": suggested}, f, indent=2)
    print(f"  -> {json_path}")

    return {"name": results_dir.name, "stats": stats, "suggested": suggested}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", action="append", required=True,
                   help="Directory containing *_log.csv files. Repeat for multiple runs.")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    summary = []
    for rd in args.results_dir:
        info = process_one(Path(rd), out_dir)
        if info is not None:
            summary.append(info)

    if summary:
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nAggregate summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
