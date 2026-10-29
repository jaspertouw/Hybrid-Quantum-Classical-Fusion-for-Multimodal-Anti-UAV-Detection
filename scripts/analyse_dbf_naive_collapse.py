"""
analyse_dbf_naive_collapse.py
=============================
Diagnose why DBF is frame-identical to naive late fusion in the Det-only runs.

Reconstructs the per-stream belief inputs to the fusion operators by joining
the single-modality (--mode ir / --mode rgb) per-frame logs, then computes,
per frame:

    C      = b_rgb_uav * b_th_bg + b_rgb_bg * b_th_uav        (conflict)
    eta    = (1 - C**lam)**(1/lam)                            (discount)
    u~     = 1 - eta * (1 - u)                                (post-discount vacuity)
    Delta  = b_uav - b_bg  per stream                         (per-stream vote)

and the three gate decisions implied by run_detector_only.py
(`b_uav > b_bg`; the theta_cls clause is checked separately):

    naive : Delta_rgb + Delta_th            > 0
    ABF   : u_th * Delta_rgb + u_rgb * Delta_th   > 0
    DBF   : u~_th * Delta_rgb + u~_rgb * Delta_th > 0

The hypothesis under test: DBF == naive on every frame because
(a) on stream-agreement frames all operators trivially coincide, and
(b) on disagreement frames C >> u, so u~_rgb ~= u~_th and DBF's vacuity
    weighting collapses to equal weights (= naive), while ABF's raw
    u_th/u_rgb ratio survives and diverges.

Optionally cross-checks the reconstruction against the dual-mode logs
(abf/dbf/naive results dirs): recomputes u_fused / b_uav / b_bg and compares
to the logged values, and counts frames where the theta_cls clause would bind.

Usage (paths are results dirs containing <seq>_log.csv):
    python scripts/analyse_dbf_naive_collapse.py \
        --ir_dir  results/detonly_ir_seed0 \
        --rgb_dir results/detonly_rgb_seed0 \
        [--abf_dir results/detonly_abf_seed0] \
        [--dbf_dir results/detonly_dbf_seed0] \
        [--naive_dir results/detonly_naive_seed0] \
        [--lam 1.0] [--theta_cls 1.0] [--out summary.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

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
    df = df.rename(columns={"u": f"u_{tag}", "b_uav": f"buav_{tag}"})
    return df


def load_dual_logs(results_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, "*_log.csv")))
    if not paths:
        raise SystemExit(f"No *_log.csv in {results_dir}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p, usecols=["frame", "u_fused", "b_uav", "b_bg"])
        df["seq"] = os.path.basename(p)[: -len("_log.csv")]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def describe(x: np.ndarray, name: str) -> dict:
    x = x[np.isfinite(x)]
    d = {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }
    print(f"  {name:<22s} mean={d['mean']:.4f}  median={d['median']:.4f}  "
          f"p95={d['p95']:.4f}  p99={d['p99']:.4f}  max={d['max']:.4f}  (n={d['n']})")
    return d


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir_dir",   required=True)
    ap.add_argument("--rgb_dir",  required=True)
    ap.add_argument("--abf_dir",  default=None)
    ap.add_argument("--dbf_dir",  default=None)
    ap.add_argument("--naive_dir", default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--miss_convention", choices=["dogmatic_bg", "vacuous"],
                    default="dogmatic_bg",
                    help="How a no-detection frame's opinion is encoded. "
                         "'dogmatic_bg' matches the pipeline's _ZERO_UNC "
                         "placeholder (b=(0,1), u=1; tracker.py); 'vacuous' "
                         "uses the partition-consistent b=(0,0), u=1.")
    ap.add_argument("--theta_cls", type=float, default=1.0,
                    help="theta used at test time; reports how often the "
                         "u_fused < theta clause would bind")
    ap.add_argument("--out", default=None, help="optional JSON summary path")
    args = ap.parse_args()

    ir  = load_single_logs(args.ir_dir,  "th")
    rgb = load_single_logs(args.rgb_dir, "rgb")
    df = ir.merge(rgb, on=["seq", "frame"], how="inner")
    n_total = len(df)
    df = df.dropna(subset=["u_th", "buav_th", "u_rgb", "buav_rgb"])
    print(f"Joined {n_total} frames ({len(df)} with beliefs on both streams) "
          f"from {df['seq'].nunique()} sequences.")

    # Partition of unity: b_bg = 1 - u - b_uav  (clip for numerical safety)
    df["bbg_th"]  = (1.0 - df["u_th"]  - df["buav_th"]).clip(0.0, 1.0)
    df["bbg_rgb"] = (1.0 - df["u_rgb"] - df["buav_rgb"]).clip(0.0, 1.0)

    # Miss frames: _yolo_run returns _ZERO_UNC (u=1, b_uav=0) when NMS yields
    # no detection.  Under the pipeline convention the placeholder belief is
    # the dogmatic background opinion b=(0,1) — NOT the partition-consistent
    # vacuous b=(0,0) — so overwrite b_bg accordingly.
    df["miss_th"]  = (df["u_th"]  >= 1.0 - 1e-9) & (df["buav_th"]  <= 1e-12)
    df["miss_rgb"] = (df["u_rgb"] >= 1.0 - 1e-9) & (df["buav_rgb"] <= 1e-12)
    if args.miss_convention == "dogmatic_bg":
        df.loc[df["miss_th"],  "bbg_th"]  = 1.0
        df.loc[df["miss_rgb"], "bbg_rgb"] = 1.0
    n_miss_one  = int((df["miss_th"] ^ df["miss_rgb"]).sum())
    n_miss_both = int((df["miss_th"] & df["miss_rgb"]).sum())
    print(f"Miss frames: one-stream={n_miss_one}  both={n_miss_both}  "
          f"(convention: {args.miss_convention})")

    # Conflict, discount, post-discount vacuity
    df["C"] = (df["buav_rgb"] * df["bbg_th"]
               + df["bbg_rgb"] * df["buav_th"]).clip(0.0, 1.0)
    eta = (1.0 - df["C"] ** args.lam) ** (1.0 / args.lam)
    df["eta"]    = eta
    df["ut_th"]  = 1.0 - eta * (1.0 - df["u_th"])
    df["ut_rgb"] = 1.0 - eta * (1.0 - df["u_rgb"])

    # Per-stream votes and the three implied gate decisions
    df["d_th"]  = df["buav_th"]  - df["bbg_th"]
    df["d_rgb"] = df["buav_rgb"] - df["bbg_rgb"]
    df["dec_naive"] = (df["d_rgb"] + df["d_th"]) > 0
    df["dec_abf"]   = (df["u_th"]  * df["d_rgb"] + df["u_rgb"]  * df["d_th"]) > 0
    df["dec_dbf"]   = (df["ut_th"] * df["d_rgb"] + df["ut_rgb"] * df["d_th"]) > 0
    df["disagree"] = np.sign(df["d_rgb"]) != np.sign(df["d_th"])

    summary = {"lam": args.lam, "n_frames": len(df),
               "miss_convention": args.miss_convention,
               "n_miss_one": n_miss_one, "n_miss_both": n_miss_both}

    print("\n--- Per-stream inputs (all frames) ---")
    summary["u_rgb"]    = describe(df["u_rgb"].to_numpy(),    "u_rgb")
    summary["u_th"]     = describe(df["u_th"].to_numpy(),     "u_th")
    summary["buav_rgb"] = describe(df["buav_rgb"].to_numpy(), "b_uav_rgb")
    summary["buav_th"]  = describe(df["buav_th"].to_numpy(),  "b_uav_th")

    print("\n--- Conflict and discount (all frames) ---")
    summary["C_all"]   = describe(df["C"].to_numpy(),   "C")
    summary["eta_all"] = describe(df["eta"].to_numpy(), "eta = 1 - C (lam=1)")

    both_fire = df[~df["miss_th"] & ~df["miss_rgb"]]
    if len(both_fire):
        print("\n--- Genuine conflict (both streams detected) ---")
        summary["C_bothfire"] = describe(both_fire["C"].to_numpy(),
                                         "C (both-fire)")

    frac_dis = float(df["disagree"].mean())
    print(f"\nStream argmax disagreement: {df['disagree'].sum()} frames "
          f"({100*frac_dis:.2f}%)")
    summary["frac_disagree"] = frac_dis

    dis = df[df["disagree"]]
    if len(dis):
        print("\n--- Disagreement frames only (where operators CAN differ) ---")
        summary["C_dis"] = describe(dis["C"].to_numpy(), "C")
        # Weight ratios: ABF uses u, DBF uses u~.  Ratio -> 1 means equal
        # weights = naive.  Report max(r, 1/r) so 1.0 is the floor.
        r_abf = (dis["u_th"]  / dis["u_rgb"].clip(lower=1e-12)).to_numpy()
        r_dbf = (dis["ut_th"] / dis["ut_rgb"].clip(lower=1e-12)).to_numpy()
        summary["abf_weight_ratio"] = describe(np.maximum(r_abf, 1.0 / r_abf),
                                               "ABF ratio max(r,1/r)")
        summary["dbf_weight_ratio"] = describe(np.maximum(r_dbf, 1.0 / r_dbf),
                                               "DBF ratio max(r,1/r)")

    print("\n--- Decision agreement across operators ---")
    for a, b in [("dbf", "naive"), ("abf", "naive"), ("dbf", "abf")]:
        diff = int((df[f"dec_{a}"] != df[f"dec_{b}"]).sum())
        diff_dis = int((dis[f"dec_{a}"] != dis[f"dec_{b}"]).sum()) if len(dis) else 0
        print(f"  {a.upper():>5s} vs {b.upper():<5s}: {diff} differing frames "
              f"({diff_dis} of them on disagreement frames)")
        summary[f"n_diff_{a}_{b}"] = diff

    # ----- Optional cross-checks against the dual-mode logs -----
    for tag, d in [("abf", args.abf_dir), ("dbf", args.dbf_dir),
                   ("naive", args.naive_dir)]:
        if d is None:
            continue
        dual = load_dual_logs(d).rename(columns={
            "u_fused": "u_fused_log", "b_uav": "buav_log", "b_bg": "bbg_log"})
        m = df.merge(dual, on=["seq", "frame"], how="inner").dropna(
            subset=["u_fused_log", "buav_log", "bbg_log"])
        dec_log = m["buav_log"] > m["bbg_log"]
        mismatch = int((m[f"dec_{tag}"] != dec_log).sum())
        binds = int((dec_log & (m["u_fused_log"] >= args.theta_cls)).sum())
        print(f"\n--- Cross-check vs {tag} dual logs ({d}) ---")
        print(f"  joined frames                          : {len(m)}")
        print(f"  reconstructed decision != logged argmax: {mismatch}")
        print(f"  frames where theta_cls clause binds    : {binds} "
              f"(argmax passes but u_fused >= {args.theta_cls})")
        summary[f"xcheck_{tag}"] = {"n": len(m), "decision_mismatch": mismatch,
                                    "theta_binds": binds}
        # Recompute the fused quantities for the exact operator and compare.
        if tag == "naive":
            buav_hat = 0.5 * (m["buav_rgb"] + m["buav_th"])
            u_hat = 1.0 - buav_hat                     # run_detector_only._fuse_naive
        else:
            if tag == "abf":
                u_r, u_t = m["u_rgb"], m["u_th"]
                bu_r, bu_t = m["buav_rgb"], m["buav_th"]
            else:  # dbf: discounted opinions into ABF
                e = m["eta"]
                u_r, u_t = m["ut_rgb"], m["ut_th"]
                bu_r, bu_t = e * m["buav_rgb"], e * m["buav_th"]
            den = (u_r + u_t).clip(lower=1e-8)
            buav_hat = (bu_r * u_t + bu_t * u_r) / den
            u_hat = 2.0 * u_r * u_t / den
        err_b = float(np.nanmax(np.abs(buav_hat - m["buav_log"])))
        err_u = float(np.nanmax(np.abs(u_hat - m["u_fused_log"])))
        print(f"  max |b_uav recon - logged|             : {err_b:.2e}")
        print(f"  max |u_fused recon - logged|           : {err_u:.2e}")
        summary[f"xcheck_{tag}"]["max_abs_err_buav"] = err_b
        summary[f"xcheck_{tag}"]["max_abs_err_ufused"] = err_u

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary -> {args.out}")


if __name__ == "__main__":
    main()
