"""
analyse_sigma_best_size_stratified.py
=====================================
Target-size-stratified sigma2_best discriminability for IR-only and RGB-only.

Tests whether IR's outcome-insensitive spatial variance (high success/failure
overlap, ~0.80, vs RGB ~0.55 in analyse_sigma_best_ir_vs_rgb_conditioned.py) is
driven by the THERMAL MODALITY or by TARGET SIZE.

Three competing hypotheses the stratification discriminates:
  - homogeneous thermal appearance : IR overlap stays high across ALL size bins,
                                      RGB stays lower across all bins.
  - size effect                    : BOTH modalities high overlap in the small
                                      bin, lower in larger bins.
  - resolution                     : IR overlap worsens specifically in the small
                                      bin, more than RGB's does.

Frame selection (identical to analyse_sigma_best_ir_vs_rgb_conditioned.py)
--------------------------------------------------------------------------
  present (modality exist flag) AND detected (IoU > 0).
  success = IoU >= success_iou (0.5); failure = 0 < IoU < success_iou.

Stratification
--------------
  normalized target size = (pred_w * pred_h) / frame_area, using the PREDICTED
  box (what the Gaussian head conditions on).  frame_area is taken from the
  actual decoded image (predictions live in native image coords — preprocessing
  does NOT resize, scale_boxes maps back to image_bgr.shape), auto-detected per
  config and reported; override with --ir_area / --rgb_area.

  Bins = equal-frequency tertiles (small / medium / large), edges computed
  SEPARATELY per config (modality-specific) on the pooled (seeds 0-2) detected
  subset.  Within a config the divisor is constant, so the tertile assignment
  and every per-bin overlap/AUROC are invariant to the exact frame_area; only
  the reported edge VALUES depend on it.

Seeds 0-2 are POOLED (all are --unc_agg peak: seed 0 = *_peak folder, seeds 1/2
= *_seed{N}), giving enough failures per bin for a stable overlap / AUROC.

Usage (on Snellius):
    python scripts/analyse_sigma_best_size_stratified.py \\
        --base         /home/ssuttorp/pretrained_models/edtc_fusion \\
        --dataset_root /home/ssuttorp/Anti-UAV-RGBT/val \\
        --seeds        0 1 2 \\
        --success_iou  0.5 \\
        --out_fig      outputs/sigma_best_size_stratified.png \\
        --out_dir      reports/sigma_best_size_stratified
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402

from resources.EDTC.lib.utils.eval_metrics import compute_auroc, iou_xywh  # noqa: E402

CONFIGS = [
    {"key": "ir",  "label": "IR",  "modality": "ir"},
    {"key": "rgb", "label": "RGB", "modality": "rgb"},
]
BIN_NAMES = ["small", "medium", "large"]
SUCCESS_COLOR = "#2ca02c"
FAILURE_COLOR = "#d62728"
N_BINS_HIST = 45


def folder_for(base: Path, modality: str, seed: int) -> Path:
    """Match the eval/separability jobs' irregular naming (all --unc_agg peak):
      seed 0 : val_results_detector_only_{modality}_peak
      seed N : val_results_detector_only_{modality}_seed{N}
    The bare folder is a LEGACY mean-aggregated run — never use it here."""
    suffix = "_peak" if seed == 0 else f"_seed{seed}"
    return base / f"val_results_detector_only_{modality}{suffix}"


# ---------------------------------------------------------------------------
# Loading (mirrors analyse_sigma_best_ir_vs_rgb_conditioned.py)
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


def detect_frame_area(seq_dir: Path, modality: str):
    """Decode the first available frame to get (W, H, area). None if unreadable."""
    sub = "rgb_frames" if modality == "rgb" else "ir_frames"
    jpgs = sorted(glob.glob(os.path.join(str(seq_dir), sub, "*.jpg")))
    for p in jpgs[:5]:                      # first few in case the very first is corrupt
        img = cv2.imread(p)
        if img is not None:
            h, w = img.shape[:2]
            return w, h, float(w * h)
    return None


# ---------------------------------------------------------------------------
# Per-(config) record building, pooled across seeds
# ---------------------------------------------------------------------------

def build_records(base, dataset_root, modality, seeds, area_override):
    """Return detected-subset frames pooled over seeds.

    Each record: {sigma2, iou, norm_size}.  norm_size = pred_w*pred_h/frame_area.
    Only present + detected (IoU>0) frames are kept.
    """
    records = []
    area_cache: dict[str, float] = {}
    seen_dims: dict[tuple, int] = {}
    n_present = n_no_area = 0

    for seed in seeds:
        results_dir = folder_for(base, modality, seed)
        if not results_dir.is_dir():
            print(f"  [skip] seed {seed}: missing {results_dir}")
            continue
        for log_path in sorted(results_dir.glob("*_log.csv")):
            seq = log_path.name.replace("_log.csv", "")
            if seq.endswith("_ir"):
                continue
            pred_path = results_dir / f"{seq}.txt"
            seq_dir = dataset_root / seq
            if not pred_path.exists() or not seq_dir.exists():
                continue

            rows = load_seq_log(log_path)
            if not rows:
                continue
            keys = rows[0].keys()
            sigma_col = "sigma2_best" if "sigma2_best" in keys else (
                "sigma2" if "sigma2" in keys else None)
            if sigma_col is None:
                continue
            preds = load_predictions(pred_path)
            ir_exist, ir_gt, rgb_exist, rgb_gt = load_labels(seq_dir)

            # Frame area for this sequence (constant within a sequence).
            if area_override is not None:
                area = float(area_override)
            else:
                key = str(seq_dir)
                if key not in area_cache:
                    dims = detect_frame_area(seq_dir, modality)
                    if dims is None:
                        area_cache[key] = None
                    else:
                        w, h, a = dims
                        area_cache[key] = a
                        seen_dims[(w, h)] = seen_dims.get((w, h), 0) + 1
                area = area_cache[key]
            if area is None or area <= 0:
                n_no_area += 1
                continue

            n = min(len(rows), len(preds), len(ir_exist), len(rgb_exist))
            for i in range(n):
                s = _f(rows[i].get(sigma_col))
                if s is None:
                    continue
                present = bool(rgb_exist[i]) if modality == "rgb" else bool(ir_exist[i])
                if not present:
                    continue
                n_present += 1
                gt = rgb_gt[i] if modality == "rgb" else ir_gt[i]
                if gt is None:
                    continue
                pred = preds[i]
                if not (pred[2] > 0 and pred[3] > 0):
                    continue                          # no detection
                iou = iou_xywh(pred, gt)
                if iou <= 0.0:
                    continue                          # detected but zero overlap -> excluded
                records.append({
                    "sigma2": s,
                    "iou": iou,
                    "norm_size": (pred[2] * pred[3]) / area,
                })

    return records, seen_dims, n_present, n_no_area


# ---------------------------------------------------------------------------
# Metrics (identical conventions to the previous analysis)
# ---------------------------------------------------------------------------

def overlap_coefficient(succ_log, fail_log, bins=N_BINS_HIST):
    if succ_log.size == 0 or fail_log.size == 0:
        return float("nan")
    allv = np.concatenate([succ_log, fail_log])
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    p_s, _ = np.histogram(succ_log, bins=edges)
    p_f, _ = np.histogram(fail_log, bins=edges)
    p_s = p_s / p_s.sum()
    p_f = p_f / p_f.sum()
    return float(np.minimum(p_s, p_f).sum())


def auroc_failure(sigma, failure):
    if sigma.size == 0 or len(np.unique(failure)) < 2:
        return float("nan")
    return float(compute_auroc(sigma, failure))


def bin_stats(sigma, iou, success_iou):
    """Per-bin stats for one (config, size-bin) subset."""
    succ_mask = iou >= success_iou
    fail_mask = ~succ_mask
    sig_succ = sigma[succ_mask]
    sig_fail = sigma[fail_mask]
    out = {
        "n_success": int(succ_mask.sum()),
        "n_failure": int(fail_mask.sum()),
        "mean_succ": float(sig_succ.mean()) if sig_succ.size else float("nan"),
        "mean_fail": float(sig_fail.mean()) if sig_fail.size else float("nan"),
        "overlap": float("nan"),
        "auroc": float("nan"),
        "_log_succ": np.log(sig_succ[sig_succ > 0]) if sig_succ.size else np.array([]),
        "_log_fail": np.log(sig_fail[sig_fail > 0]) if sig_fail.size else np.array([]),
    }
    if sig_succ.size and sig_fail.size:
        out["overlap"] = overlap_coefficient(out["_log_succ"], out["_log_fail"])
    if sigma.size:
        out["auroc"] = auroc_failure(sigma, fail_mask.astype(int))
    return out


def stratify_config(records, success_iou):
    sigma = np.array([r["sigma2"] for r in records], float)
    iou = np.array([r["iou"] for r in records], float)
    size = np.array([r["norm_size"] for r in records], float)

    # Equal-frequency tertiles on normalized size (modality-specific edges).
    q1, q2 = np.quantile(size, [1.0 / 3.0, 2.0 / 3.0])
    # assignment: small <= q1 < medium <= q2 < large
    bin_idx = np.where(size <= q1, 0, np.where(size <= q2, 1, 2))

    edges = [
        (float(size.min()), float(q1)),
        (float(q1), float(q2)),
        (float(q2), float(size.max())),
    ]
    bins = []
    for b in range(3):
        m = bin_idx == b
        st = bin_stats(sigma[m], iou[m], success_iou)
        st["edges"] = edges[b]
        st["n_total"] = int(m.sum())
        bins.append(st)
    return {"edges": edges, "bins": bins, "n_detected": len(records)}


# ---------------------------------------------------------------------------
# Figure: 2 rows (IR/RGB) x 3 cols (small/medium/large)
# ---------------------------------------------------------------------------

def make_figure(results, out_fig, success_iou):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    for row, cfg in enumerate(CONFIGS):
        res = results.get(cfg["key"])
        if res is None:
            for col in range(3):
                axes[row, col].axis("off")
            continue
        # common x-range across the row (per config) for comparability
        all_logs = np.concatenate(
            [np.concatenate([b["_log_succ"], b["_log_fail"]]) for b in res["bins"]
             if b["_log_succ"].size or b["_log_fail"].size])
        lo, hi = float(all_logs.min()), float(all_logs.max())
        row_edges = np.linspace(lo, hi, N_BINS_HIST + 1)
        for col, b in enumerate(res["bins"]):
            ax = axes[row, col]
            ax.hist(b["_log_succ"], bins=row_edges, alpha=0.55, color=SUCCESS_COLOR,
                    label=f"success (n={b['n_success']})")
            ax.hist(b["_log_fail"], bins=row_edges, alpha=0.55, color=FAILURE_COLOR,
                    label=f"failure (n={b['n_failure']})")
            e0, e1 = b["edges"]
            ax.set_title(
                f"{cfg['label']} — {BIN_NAMES[col]}  "
                f"(size {e0:.2e}–{e1:.2e})\n"
                f"overlap={b['overlap']:.3f}, AUROC={b['auroc']:.3f}",
                fontsize=9)
            if row == 1:
                ax.set_xlabel(r"$\log\,\hat{\sigma}^2_{\mathrm{best}}$")
            if col == 0:
                ax.set_ylabel(f"{cfg['label']}-only\ncount")
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        r"Size-stratified $\hat{\sigma}^2_{\mathrm{best}}$ by localisation outcome "
        f"(detected present frames, IoU>{0}; success≥{success_iou:g}; seeds 0–2 pooled)",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"\nWrote figure -> {out_fig}  (300 DPI)")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results, frame_dims, success_iou):
    cols = ["Config", "Size bin", "bin edges (norm)", "n_succ", "n_fail",
            "mean_succ", "mean_fail", "overlap", "AUROC"]
    widths = [8, 9, 24, 8, 8, 11, 11, 9, 8]
    line = "".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print("\n" + "=" * len(line))
    print(f"Size-stratified summary - detected present frames (IoU>0), "
          f"success_iou={success_iou}, seeds 0-2 pooled")
    for cfg in CONFIGS:
        d = frame_dims.get(cfg["key"])
        if d:
            print(f"  {cfg['label']} frame size used: {d}")
    print("=" * len(line))
    print(line)
    print("-" * len(line))
    for cfg in CONFIGS:
        res = results.get(cfg["key"])
        if res is None:
            continue
        for col, b in enumerate(res["bins"]):
            e0, e1 = b["edges"]
            vals = [cfg["label"], BIN_NAMES[col], f"{e0:.3e}-{e1:.3e}",
                    str(b["n_success"]), str(b["n_failure"]),
                    f"{b['mean_succ']:.4g}", f"{b['mean_fail']:.4g}",
                    f"{b['overlap']:.3f}", f"{b['auroc']:.3f}"]
            print("".join(f"{v:>{w}}" for v, w in zip(vals, widths)))
        print("-" * len(line))
    print("overlap = histogram overlap coefficient of log sigma2_best "
          "(success vs failure); 1=indistinguishable, 0=disjoint.")
    print("AUROC   = sigma2_best as failure predictor within the bin.")


def write_report(path, results, frame_dims, success_iou, seeds):
    L = ["# Size-stratified sigma2_best discriminability — IR vs RGB\n"]
    L.append(f"- Split: validation; seeds {seeds} (pooled).")
    L.append(f"- Subset: present + detected (IoU > 0). success = IoU ≥ {success_iou}; "
             f"failure = 0 < IoU < {success_iou}.")
    L.append("- Normalized size = pred_w·pred_h / frame_area; equal-frequency "
             "tertiles, modality-specific edges.")
    for cfg in CONFIGS:
        d = frame_dims.get(cfg["key"])
        if d:
            L.append(f"- {cfg['label']} frame size used: {d}.")
    L.append("")
    L.append("| Config | Size bin | bin edges (norm) | n_succ | n_fail | "
             "mean_succ | mean_fail | overlap | AUROC |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        res = results.get(cfg["key"])
        if res is None:
            continue
        for col, b in enumerate(res["bins"]):
            e0, e1 = b["edges"]
            L.append(
                f"| {cfg['label']} | {BIN_NAMES[col]} | {e0:.3e}–{e1:.3e} | "
                f"{b['n_success']} | {b['n_failure']} | {b['mean_succ']:.4g} | "
                f"{b['mean_fail']:.4g} | {b['overlap']:.3f} | {b['auroc']:.3f} |")
    L.append("")
    L.append("## Reading the result")
    L.append("- **Thermal-appearance hypothesis:** IR overlap stays high across "
             "all three bins; RGB stays lower across all bins.")
    L.append("- **Size effect:** both modalities show high overlap in *small* and "
             "lower overlap in *large*.")
    L.append("- **Resolution:** IR overlap worsens in the *small* bin specifically, "
             "more than RGB's does.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Wrote report -> {path}")


def to_json_safe(results):
    out = {}
    for k, res in results.items():
        if res is None:
            out[k] = None
            continue
        bins = []
        for col, b in enumerate(res["bins"]):
            bins.append({kk: vv for kk, vv in b.items()
                         if not kk.startswith("_")} | {"bin": BIN_NAMES[col]})
        out[k] = {"edges": res["edges"], "n_detected": res["n_detected"], "bins": bins}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--success_iou", type=float, default=0.5)
    ap.add_argument("--ir_area", type=float, default=None,
                    help="Override IR frame area (px^2). Default: auto-detect from images.")
    ap.add_argument("--rgb_area", type=float, default=None,
                    help="Override RGB frame area (px^2). Default: auto-detect from images.")
    ap.add_argument("--out_fig", type=Path,
                    default=_REPO_ROOT / "outputs" / "sigma_best_size_stratified.png")
    ap.add_argument("--out_dir", type=Path,
                    default=_REPO_ROOT / "reports" / "sigma_best_size_stratified")
    args = ap.parse_args()

    print(f"Base         : {args.base}")
    print(f"Dataset root : {args.dataset_root}")
    print(f"Seeds        : {args.seeds}   success_iou={args.success_iou}")

    results = {}
    frame_dims = {}
    for cfg in CONFIGS:
        print(f"\n=== {cfg['label']}-only ({cfg['modality']}) ===")
        override = args.ir_area if cfg["modality"] == "ir" else args.rgb_area
        records, seen_dims, n_present, n_no_area = build_records(
            args.base, args.dataset_root, cfg["modality"], args.seeds, override)
        if override is not None:
            frame_dims[cfg["key"]] = f"override area={override:.0f} px^2"
        elif seen_dims:
            dom = max(seen_dims.items(), key=lambda kv: kv[1])
            frame_dims[cfg["key"]] = (
                f"{dom[0][0]}x{dom[0][1]} (area={dom[0][0]*dom[0][1]} px^2)"
                + ("" if len(seen_dims) == 1
                   else f"; {len(seen_dims)} distinct sizes seen: {dict(seen_dims)}"))
        print(f"  present={n_present}  detected(IoU>0)={len(records)}  "
              f"frames_dropped_no_area={n_no_area}")
        if len(records) < 30:
            print("  [skip] too few detected frames")
            results[cfg["key"]] = None
            continue
        res = stratify_config(records, args.success_iou)
        results[cfg["key"]] = res
        for col, b in enumerate(res["bins"]):
            print(f"  {BIN_NAMES[col]:7s} edges=({b['edges'][0]:.3e},{b['edges'][1]:.3e}) "
                  f"n=({b['n_success']}+{b['n_failure']})  "
                  f"overlap={b['overlap']:.3f}  AUROC={b['auroc']:.3f}")

    make_figure(results, args.out_fig, args.success_iou)
    print_table(results, frame_dims, args.success_iou)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "sigma_best_size_stratified.json", "w") as f:
        json.dump({"frame_dims": frame_dims, "success_iou": args.success_iou,
                   "seeds": args.seeds, "configs": to_json_safe(results)}, f, indent=2)
    print(f"Wrote JSON   -> {args.out_dir / 'sigma_best_size_stratified.json'}")
    write_report(args.out_dir / "sigma_best_size_stratified_report.md",
                 results, frame_dims, args.success_iou, args.seeds)


if __name__ == "__main__":
    main()
