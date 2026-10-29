"""
analyse_sigma_best_ir_vs_rgb_conditioned.py
============================================
Per-frame spatial-uncertainty (sigma2_best) distributions for the IR-only and
RGB-only detectors, conditioned on localisation success vs. failure.

Motivation
----------
Table 5 reports sigma2_best as a near-chance failure detector for IR
(AUROC ~ 0.560) but a strong one for RGB (AUROC ~ 0.748), even though IR is
competitive on localisation (AP@0.5 0.944 vs 0.907).  This script isolates the
*genuine* localisation-uncertainty signal — restricting to frames where the
stream actually emitted a detection that overlapped the GT — and characterises
how flat / discriminative the success vs failure sigma2_best distributions are
for each modality.

Frame selection (the requested "conditioned-on-detection" subset)
-----------------------------------------------------------------
  present       : ground-truth visibility flag == 1 (modality-specific exist[i]).
  detected      : a non-vacuous detection was emitted AND it overlapped the GT,
                  i.e. IoU > 0.  A YOLO miss logs the _ZERO_UNC placeholder
                  (sigma2 = 1e6, pred box = [0,0,0,0] -> IoU = 0); such frames
                  carry a vacuous opinion and no meaningful spatial uncertainty,
                  so they are EXCLUDED.
  success       : IoU >= success_iou (default 0.5).
  failure       : 0 < IoU < success_iou  (detected something, localised poorly).

AUROC sanity check vs. Table 5
------------------------------
The Table-5 separability number is produced by
analyse_uncertainty_separability_detonly.py, which conditions on
argmax(b_fused) == UAV and keeps IoU == 0 detected failures.  That convention
is reproduced verbatim in the `_table5_auroc` block so the pipeline can be
sanity-checked against the paper (~0.560 IR / ~0.748 RGB).  It is reported
ALONGSIDE the requested detected-subset AUROC, never silently substituted.

Aggregation
-----------
Every scalar (mean / std of sigma2_best, overlap coefficient, AUROC) is computed
per seed and then averaged across seeds (mean reported; std-across-seeds shown
in the report for context).  The overlaid histogram pools all seeds' frames.

Usage (on Snellius):
    python scripts/analyse_sigma_best_ir_vs_rgb_conditioned.py \\
        --base         /home/ssuttorp/pretrained_models/edtc_fusion \\
        --dataset_root /home/ssuttorp/Anti-UAV-RGBT/val \\
        --seeds        0 1 2 \\
        --success_iou  0.5 \\
        --out_fig      outputs/sigma_best_ir_vs_rgb_conditioned.png \\
        --out_dir      reports/sigma_best_ir_vs_rgb_conditioned
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from resources.EDTC.lib.utils.eval_metrics import compute_auroc, iou_xywh

# Per-seed result-folder naming (mirrors analyse_separability_detonly_seeds.job
# and print_metrics_tables.py::_seed_runs): seed 0 has no suffix, seeds 1/2 add
# "_seed{N}".  IR/RGB single-modality folders drop the "_peak" tag.
CONFIGS = [
    {"key": "ir",  "label": "IR",  "modality": "ir",  "color_axis": "C0"},
    {"key": "rgb", "label": "RGB", "modality": "rgb", "color_axis": "C1"},
]
SUCCESS_COLOR = "#2ca02c"   # green
FAILURE_COLOR = "#d62728"   # red
N_BINS = 50


def folder_for(base: Path, modality: str, seed: int) -> Path:
    """Per-seed result folder, matching the eval/separability jobs' irregular
    naming so we read the SAME peak-aggregated logs Table 5 used:

      seed 0 : val_results_detector_only_{modality}_peak     (eval_detector_only_{ir,rgb}.job)
      seed N : val_results_detector_only_{modality}_seed{N}  (eval_detector_only_{ir,rgb}_seeds.job)

    Both are produced with --unc_agg peak. The bare 'val_results_detector_only_ir'
    folder is a LEGACY --unc_agg mean run (different sigma2 signal) and must NOT
    be used here — mixing mean/peak across seeds is invalid and was the cause of
    the inverted seed-0 RGB AUROC.
    """
    suffix = "_peak" if seed == 0 else f"_seed{seed}"
    return base / f"val_results_detector_only_{modality}{suffix}"


# ---------------------------------------------------------------------------
# Loading (mirrors analyse_uncertainty_separability_detonly.py verbatim)
# ---------------------------------------------------------------------------

def load_seq_log(csv_path: Path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def load_predictions(txt_path: Path):
    preds = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                preds.append([0.0, 0.0, 0.0, 0.0])
                continue
            parts = line.replace("\t", ",").split(",")
            preds.append([float(p) for p in parts[:4]])
    return preds


def load_labels(seq_dir: Path):
    with open(seq_dir / "IR_label.json") as f:
        ir = json.load(f)
    with open(seq_dir / "visible_label.json") as f:
        rgb = json.load(f)
    return ir["exist"], ir["gt_rect"], rgb["exist"], rgb["gt_rect"]


def _f(v):
    if v in (None, "", "None"):
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Per-(config, seed) per-frame record building
# ---------------------------------------------------------------------------

def build_records(results_dir: Path, dataset_root: Path, modality: str):
    """Return per-present-frame dicts: sigma2_best, b_uav, u, iou, detected.

    Every PRESENT frame is returned (no IoU/detection filtering here) so the
    caller can apply both the requested detected-subset filter and the Table-5
    convention to the same underlying frames.
    """
    records = []
    n_present = 0
    for log_path in sorted(results_dir.glob("*_log.csv")):
        seq = log_path.name.replace("_log.csv", "")
        if seq.endswith("_ir"):          # skip auxiliary *_ir companion logs
            continue
        pred_path = results_dir / f"{seq}.txt"
        seq_dir = dataset_root / seq
        if not pred_path.exists() or not seq_dir.exists():
            continue

        rows = load_seq_log(log_path)
        if not rows:
            continue
        keys = rows[0].keys()
        u_col = "u_fused" if "u_fused" in keys else ("u" if "u" in keys else None)
        sigma_col = "sigma2_best" if "sigma2_best" in keys else (
            "sigma2" if "sigma2" in keys else None)
        if u_col is None or sigma_col is None:
            print(f"[warn] {log_path.name}: no u/sigma column, skipping")
            continue
        has_bbg = "b_bg" in keys

        preds = load_predictions(pred_path)
        ir_exist, ir_gt, rgb_exist, rgb_gt = load_labels(seq_dir)

        n = min(len(rows), len(preds), len(ir_exist), len(rgb_exist))
        for i in range(n):
            r = rows[i]
            s = _f(r.get(sigma_col))
            u = _f(r.get(u_col))
            b_uav = _f(r.get("b_uav"))
            if s is None:
                continue                          # no spatial uncertainty logged

            present = bool(rgb_exist[i]) if modality == "rgb" else bool(ir_exist[i])
            if not present:
                continue
            n_present += 1

            gt = rgb_gt[i] if modality == "rgb" else ir_gt[i]
            if gt is None:
                continue
            pred = preds[i]
            detected = (pred[2] > 0 and pred[3] > 0)
            iou = iou_xywh(pred, gt) if detected else 0.0

            # b_bg: logged column when present (correct for fusion logs); else
            # derive from the K=2 partition (valid for single-modality logs).
            b_bg = _f(r.get("b_bg")) if has_bbg else (
                None if (b_uav is None or u is None) else 1.0 - b_uav - u)

            records.append({
                "sigma2_best": s,
                "b_uav": b_uav,
                "b_bg": b_bg,
                "iou": iou,
                "detected": detected,
            })
    return records, n_present


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def overlap_coefficient(succ_log: np.ndarray, fail_log: np.ndarray,
                        bins: int = N_BINS):
    """Histogram overlap coefficient OVL = sum_i min(p_succ_i, p_fail_i).

    Computed on a SHARED set of equal-width bins spanning the pooled range of
    the (already log-transformed) success and failure samples.  Returns a value
    in [0, 1]: 1.0 = identical distributions (signal carries no information
    about success/failure), 0.0 = perfectly disjoint.  Invariant to the log
    base (a monotone rescale of the axis).
    """
    if succ_log.size == 0 or fail_log.size == 0:
        return float("nan")
    allv = np.concatenate([succ_log, fail_log])
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    p_s, _ = np.histogram(succ_log, bins=edges, density=False)
    p_f, _ = np.histogram(fail_log, bins=edges, density=False)
    p_s = p_s / p_s.sum()
    p_f = p_f / p_f.sum()
    return float(np.minimum(p_s, p_f).sum())


def auroc_failure(sigma: np.ndarray, failure: np.ndarray):
    """AUROC of sigma2_best as a FAILURE predictor (higher sigma -> failure).

    Positive label = failure, score = sigma2_best — identical sign convention
    to analyse_uncertainty_separability_detonly.py / compute_auroc.
    """
    if sigma.size == 0 or len(np.unique(failure)) < 2:
        return float("nan")
    return float(compute_auroc(sigma, failure))


def per_seed_stats(records, success_iou):
    """Compute the requested detected-subset stats + the Table-5 sanity AUROC
    for one (config, seed)."""
    sigma = np.array([r["sigma2_best"] for r in records], float)
    iou = np.array([r["iou"] for r in records], float)
    b_uav = np.array([r["b_uav"] if r["b_uav"] is not None else np.nan
                      for r in records], float)
    b_bg = np.array([r["b_bg"] if r["b_bg"] is not None else np.nan
                     for r in records], float)

    # ── Requested detected subset: IoU > 0 (detected + overlapped) ──────────
    det = iou > 0.0
    sig_d = sigma[det]
    iou_d = iou[det]
    succ_mask = iou_d >= success_iou
    fail_mask = ~succ_mask                       # subset is IoU>0, so this is 0<IoU<thr
    sig_succ = sig_d[succ_mask]
    sig_fail = sig_d[fail_mask]
    failure_d = fail_mask.astype(int)

    out = {
        "n_present": len(records),
        "n_detected": int(det.sum()),
        "n_success": int(succ_mask.sum()),
        "n_failure": int(fail_mask.sum()),
        "mean_succ": float(sig_succ.mean()) if sig_succ.size else float("nan"),
        "mean_fail": float(sig_fail.mean()) if sig_fail.size else float("nan"),
        "std_succ": float(sig_succ.std(ddof=0)) if sig_succ.size else float("nan"),
        "std_fail": float(sig_fail.std(ddof=0)) if sig_fail.size else float("nan"),
        "overlap": float("nan"),
        "auroc_detected": float("nan"),
        "auroc_table5": float("nan"),
        # raw samples for pooled plotting
        "_sig_succ": sig_succ,
        "_sig_fail": sig_fail,
    }
    if sig_succ.size and sig_fail.size:
        out["overlap"] = overlap_coefficient(
            np.log(sig_succ[sig_succ > 0]), np.log(sig_fail[sig_fail > 0]))
    if sig_d.size:
        out["auroc_detected"] = auroc_failure(sig_d, failure_d)

    # ── Table-5 reproduction: condition argmax(b)==UAV, keep IoU==0 ─────────
    # Mirrors analyse_uncertainty_separability_detonly.py (conditioned default).
    valid = ~np.isnan(b_uav) & ~np.isnan(b_bg)
    cond = valid & (b_uav > b_bg)                # argmax(b_fused) == UAV
    sig_c = sigma[cond]
    fail_c = (iou[cond] < success_iou).astype(int)
    out["auroc_table5"] = auroc_failure(sig_c, fail_c)
    out["n_table5"] = int(cond.sum())
    return out


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def _mean_std(vals):
    a = np.array([v for v in vals if v is not None and not math.isnan(v)], float)
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))


def aggregate_config(base, dataset_root, modality, seeds, success_iou):
    per_seed = []
    pooled_succ, pooled_fail = [], []
    for seed in seeds:
        d = folder_for(base, modality, seed)
        if not d.is_dir():
            print(f"  [skip] seed {seed}: missing {d}")
            continue
        recs, n_present = build_records(d, dataset_root, modality)
        st = per_seed_stats(recs, success_iou)
        st["seed"] = seed
        per_seed.append(st)
        pooled_succ.append(st.pop("_sig_succ"))
        pooled_fail.append(st.pop("_sig_fail"))
        print(f"  seed {seed}: present={st['n_present']} detected={st['n_detected']} "
              f"succ={st['n_success']} fail={st['n_failure']}  "
              f"AUROC(det)={st['auroc_detected']:.3f} "
              f"AUROC(T5)={st['auroc_table5']:.3f} overlap={st['overlap']:.3f}")

    if not per_seed:
        return None

    agg = {"per_seed": per_seed}
    for field in ("mean_succ", "mean_fail", "std_succ", "std_fail",
                  "overlap", "auroc_detected", "auroc_table5"):
        m, s = _mean_std([ps[field] for ps in per_seed])
        agg[field] = m
        agg[field + "_seedstd"] = s
    agg["pooled_succ"] = (np.concatenate(pooled_succ)
                          if pooled_succ else np.array([]))
    agg["pooled_fail"] = (np.concatenate(pooled_fail)
                          if pooled_fail else np.array([]))
    agg["n_detected_total"] = int(sum(ps["n_detected"] for ps in per_seed))
    return agg


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(results, out_fig: Path, success_iou: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, cfg in zip(axes, CONFIGS):
        agg = results.get(cfg["key"])
        if agg is None:
            ax.set_title(f"{cfg['label']} — no data")
            ax.axis("off")
            continue
        succ = agg["pooled_succ"]
        fail = agg["pooled_fail"]
        succ = succ[succ > 0]
        fail = fail[fail > 0]
        log_succ = np.log(succ)
        log_fail = np.log(fail)
        allv = np.concatenate([log_succ, log_fail]) if (log_succ.size and log_fail.size) \
            else (log_succ if log_succ.size else log_fail)
        edges = np.linspace(float(allv.min()), float(allv.max()), N_BINS + 1)
        ax.hist(log_succ, bins=edges, alpha=0.55, color=SUCCESS_COLOR,
                label=f"success (IoU≥{success_iou:g}, n={log_succ.size})")
        ax.hist(log_fail, bins=edges, alpha=0.55, color=FAILURE_COLOR,
                label=f"failure (0<IoU<{success_iou:g}, n={log_fail.size})")
        ax.set_xlabel(r"$\log\,\hat{\sigma}^2_{\mathrm{best}}$")
        ax.set_ylabel("count")
        ax.set_title(
            f"{cfg['label']}-only  "
            f"(AUROC={agg['auroc_detected']:.3f}, overlap={agg['overlap']:.3f})")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        r"Spatial uncertainty $\hat{\sigma}^2_{\mathrm{best}}$ by localisation outcome "
        "(detected present frames; seeds 0–2 pooled)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"\nWrote figure -> {out_fig}  (300 DPI)")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results, success_iou):
    # ASCII-only console output (SLURM .out is often viewed as cp1252; Unicode
    # sigma/em-dash render as mojibake there). The .md report keeps Unicode.
    cols = ["Config", "mean s2(succ)", "mean s2(fail)", "std s2(succ)",
            "std s2(fail)", "overlap", "AUROC", "AUROC(T5)"]
    widths = [8, 15, 15, 14, 14, 9, 9, 11]
    line = "".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print("\n" + "=" * len(line))
    print(f"Summary - detected present frames (IoU>0), success_iou={success_iou}, "
          f"mean across seeds")
    print("=" * len(line))
    print(line)
    print("-" * len(line))
    for cfg in CONFIGS:
        agg = results.get(cfg["key"])
        if agg is None:
            continue
        vals = [cfg["label"],
                f"{agg['mean_succ']:.4g}", f"{agg['mean_fail']:.4g}",
                f"{agg['std_succ']:.4g}", f"{agg['std_fail']:.4g}",
                f"{agg['overlap']:.3f}", f"{agg['auroc_detected']:.3f}",
                f"{agg['auroc_table5']:.3f}"]
        print("".join(f"{v:>{w}}" for v, w in zip(vals, widths)))
    print("=" * len(line))
    print("AUROC      = sigma2_best as failure predictor on the detected "
          "subset (IoU>0; success vs 0<IoU<thr).")
    print("AUROC(T5)  = Table-5 convention (argmax(b)==UAV, IoU==0 failures "
          "kept) - should reproduce ~0.560 IR / ~0.748 RGB.")


def write_report(path: Path, results, success_iou, seeds, args):
    L = []
    L.append("# sigma2_best — IR vs RGB, conditioned on localisation outcome\n")
    L.append(f"- Split: validation; seeds {seeds} (mean across seeds).")
    L.append(f"- Subset: present frames with a detection that overlapped the GT "
             f"(IoU > 0). YOLO misses (IoU=0, sigma2=1e6 placeholder) excluded.")
    L.append(f"- success = IoU ≥ {success_iou}; failure = 0 < IoU < {success_iou}.")
    L.append("")
    L.append("| Config | mean σ²(succ) | mean σ²(fail) | std σ²(succ) | std σ²(fail) "
             "| overlap | AUROC | AUROC (T5 sanity) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        agg = results.get(cfg["key"])
        if agg is None:
            continue
        L.append(
            f"| {cfg['label']} | {agg['mean_succ']:.4g} | {agg['mean_fail']:.4g} "
            f"| {agg['std_succ']:.4g} | {agg['std_fail']:.4g} "
            f"| {agg['overlap']:.3f} | {agg['auroc_detected']:.3f} "
            f"| {agg['auroc_table5']:.3f} |")
    L.append("")
    L.append("**Columns.** `overlap` is the histogram overlap coefficient of "
             "log σ²_best (success vs failure); 1.0 = indistinguishable "
             "distributions, 0.0 = disjoint — it quantifies how flat / "
             "non-discriminative the spatial uncertainty is. `AUROC` is "
             "σ²_best as a failure predictor on the detected subset. "
             "`AUROC (T5 sanity)` reproduces the Table-5 separability "
             "convention (conditioned on argmax(b)==UAV, IoU=0 detected "
             "failures retained) for cross-checking against the paper "
             "(~0.560 IR / ~0.748 RGB).")
    L.append("")
    L.append("## Per-seed detail")
    for cfg in CONFIGS:
        agg = results.get(cfg["key"])
        if agg is None:
            continue
        L.append(f"\n### {cfg['label']}")
        for f in ("mean_succ", "mean_fail", "std_succ", "std_fail",
                  "overlap", "auroc_detected", "auroc_table5"):
            L.append(f"- {f}: {agg[f]:.4g}  (±{agg[f + '_seedstd']:.4g} across seeds)")
        L.append("- per-seed:")
        for ps in agg["per_seed"]:
            L.append(f"  - seed {ps['seed']}: detected={ps['n_detected']} "
                     f"(succ={ps['n_success']}, fail={ps['n_failure']}), "
                     f"AUROC(det)={ps['auroc_detected']:.3f}, "
                     f"AUROC(T5)={ps['auroc_table5']:.3f}, "
                     f"overlap={ps['overlap']:.3f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Wrote report -> {path}")


def to_json_safe(results):
    out = {}
    for k, agg in results.items():
        if agg is None:
            out[k] = None
            continue
        d = {kk: vv for kk, vv in agg.items()
             if kk not in ("pooled_succ", "pooled_fail")}
        out[k] = d
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path,
                    help="dir holding val_results_detector_only_{ir,rgb}[_seedN]")
    ap.add_argument("--dataset_root", required=True, type=Path,
                    help="Anti-UAV-RGBT/val (per-sequence label dirs)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--success_iou", type=float, default=0.5)
    ap.add_argument("--out_fig", type=Path,
                    default=_REPO_ROOT / "outputs" / "sigma_best_ir_vs_rgb_conditioned.png")
    ap.add_argument("--out_dir", type=Path,
                    default=_REPO_ROOT / "reports" / "sigma_best_ir_vs_rgb_conditioned")
    args = ap.parse_args()

    print(f"Base         : {args.base}")
    print(f"Dataset root : {args.dataset_root}")
    print(f"Seeds        : {args.seeds}   success_iou={args.success_iou}")

    results = {}
    for cfg in CONFIGS:
        print(f"\n=== {cfg['label']}-only ({cfg['modality']}) ===")
        results[cfg["key"]] = aggregate_config(
            args.base, args.dataset_root, cfg["modality"],
            args.seeds, args.success_iou)

    make_figure(results, args.out_fig, args.success_iou)
    print_table(results, args.success_iou)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "sigma_best_ir_vs_rgb.json", "w") as f:
        json.dump(to_json_safe(results), f, indent=2)
    print(f"Wrote JSON   -> {args.out_dir / 'sigma_best_ir_vs_rgb.json'}")
    write_report(args.out_dir / "sigma_best_ir_vs_rgb_report.md",
                 results, args.success_iou, args.seeds, args)


if __name__ == "__main__":
    main()
