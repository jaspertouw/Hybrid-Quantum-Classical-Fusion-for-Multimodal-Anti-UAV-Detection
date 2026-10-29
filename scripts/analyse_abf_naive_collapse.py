"""
analyse_abf_naive_collapse.py
=============================
On BOTH-DETECT frames only, quantify whether ABF (uncertainty-weighted belief
fusion) collapses numerically onto naive 0.5/0.5 averaging.

Per-stream beliefs/uncertainties are not stored in a single log; they are
reconstructed by inner-joining the single-modality detector-only logs
(``--mode ir`` and ``--mode rgb``) on (seq, frame), exactly as
``analyse_dbf_naive_collapse.py`` does:

    IR  log  -> b_th  = b_uav,  u_th  = u
    RGB log  -> b_rgb = b_uav,  u_rgb = u

A stream "missed" a frame when its row is the _ZERO_UNC placeholder
(u >= 1 - 1e-9 and b_uav <= 1e-12).  Both-detect = neither stream missed.

On the both-detect subset, for each frame:

    b_naive = 0.5 * (b_rgb + b_th)
    b_abf   = (b_rgb * u_th + b_th * u_rgb) / (u_rgb + u_th)

and the background-belief counterparts (partition of unity, b_bg = 1 - u - b_uav):

    bbg_naive = 0.5 * (bbg_rgb + bbg_th)
    bbg_abf   = (bbg_rgb * u_th + bbg_th * u_rgb) / (u_rgb + u_th)

Reports:
  1. max / mean / p99 of |b_abf - b_naive|
  2. max / mean of |u_rgb - u_th|
  3. mean of b_rgb and b_th  (belief-saturation check)
  4. # frames where the fused argmax (UAV vs bg) flips between the two rules

Usage (paths are results dirs containing <seq>_log.csv):
    python scripts/analyse_abf_naive_collapse.py \
        --ir_dir  /home/ssuttorp/pretrained_models/edtc_fusion/val_results_detector_only_ir_peak \
        --rgb_dir /home/ssuttorp/pretrained_models/edtc_fusion/val_results_detector_only_rgb_peak \
        [--out summary.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

_EPS = 1e-8  # matches belief_fusion.abf denominator guard


def load_single_logs(results_dir: str, tag: str) -> pd.DataFrame:
    """Load all <seq>_log.csv from a single-modality results dir."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*_log.csv")))
    if not paths:
        raise SystemExit(f"No *_log.csv in {results_dir}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p, usecols=["frame", "u", "b_uav"])
        df["seq"] = os.path.basename(p)[: -len("_log.csv")]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df.rename(columns={"u": f"u_{tag}", "b_uav": f"buav_{tag}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir_dir", required=True, help="single-modality IR (thermal) results dir")
    ap.add_argument("--rgb_dir", required=True, help="single-modality RGB results dir")
    ap.add_argument("--out", default=None, help="optional JSON summary path")
    args = ap.parse_args()

    ir = load_single_logs(args.ir_dir, "th")
    rgb = load_single_logs(args.rgb_dir, "rgb")
    df = ir.merge(rgb, on=["seq", "frame"], how="inner")
    n_total = len(df)
    df = df.dropna(subset=["u_th", "buav_th", "u_rgb", "buav_rgb"])
    print(f"Joined {n_total} frames ({len(df)} with beliefs on both streams) "
          f"from {df['seq'].nunique()} sequences.")

    # Miss = _ZERO_UNC placeholder (no detection emitted by that stream).
    miss_th = (df["u_th"] >= 1.0 - 1e-9) & (df["buav_th"] <= 1e-12)
    miss_rgb = (df["u_rgb"] >= 1.0 - 1e-9) & (df["buav_rgb"] <= 1e-12)
    both = df[~miss_th & ~miss_rgb].copy()
    print(f"Both-detect frames (neither stream missed): {len(both)} "
          f"of {len(df)} ({100.0 * len(both) / max(len(df), 1):.2f}%)")
    if both.empty:
        raise SystemExit("No both-detect frames; nothing to compare.")

    b_rgb, b_th = both["buav_rgb"].to_numpy(), both["buav_th"].to_numpy()
    u_rgb, u_th = both["u_rgb"].to_numpy(), both["u_th"].to_numpy()
    # Background belief per stream via partition of unity.
    bbg_rgb = np.clip(1.0 - u_rgb - b_rgb, 0.0, 1.0)
    bbg_th = np.clip(1.0 - u_th - b_th, 0.0, 1.0)

    denom = np.clip(u_rgb + u_th, _EPS, None)
    b_naive = 0.5 * (b_rgb + b_th)
    b_abf = (b_rgb * u_th + b_th * u_rgb) / denom
    bbg_naive = 0.5 * (bbg_rgb + bbg_th)
    bbg_abf = (bbg_rgb * u_th + bbg_th * u_rgb) / denom

    db = np.abs(b_abf - b_naive)
    du = np.abs(u_rgb - u_th)
    # Fused argmax (UAV vs background) under each rule.
    dec_naive = b_naive > bbg_naive
    dec_abf = b_abf > bbg_abf
    n_flip = int(np.sum(dec_naive != dec_abf))

    print("\n=== ABF vs naive on both-detect frames "
          f"(n = {len(both)}) ===")
    print("1) |b_abf - b_naive| (UAV belief):")
    print(f"     max  = {db.max():.6e}")
    print(f"     mean = {db.mean():.6e}")
    print(f"     p99  = {np.percentile(db, 99):.6e}")
    print("2) |u_rgb - u_th|:")
    print(f"     max  = {du.max():.6e}")
    print(f"     mean = {du.mean():.6e}")
    print("3) belief saturation:")
    print(f"     mean b_rgb = {b_rgb.mean():.6f}")
    print(f"     mean b_th  = {b_th.mean():.6f}")
    print("4) argmax flips (UAV<->bg between naive and ABF):")
    print(f"     n_flip = {n_flip}  ({100.0 * n_flip / len(both):.3f}% of both-detect)")

    if args.out:
        summary = {
            "n_joined": n_total,
            "n_both_detect": int(len(both)),
            "abs_b_abf_minus_b_naive": {
                "max": float(db.max()),
                "mean": float(db.mean()),
                "p99": float(np.percentile(db, 99)),
            },
            "abs_u_rgb_minus_u_th": {
                "max": float(du.max()),
                "mean": float(du.mean()),
            },
            "mean_b_rgb": float(b_rgb.mean()),
            "mean_b_th": float(b_th.mean()),
            "n_argmax_flip": n_flip,
        }
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary -> {args.out}")


if __name__ == "__main__":
    main()
