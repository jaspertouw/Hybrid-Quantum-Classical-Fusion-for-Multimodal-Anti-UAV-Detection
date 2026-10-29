"""
analyse_uncertainty_separability_detonly.py
============================================
Detector-only counterpart to analyse_uncertainty_separability_conditioned.py.

The detector-only variants have NO DETECT/TRACK state machine — there is a
single detection decision per frame — so this script drops the per-state split
and reports one failure-detection AUROC per signal (u_fused, b_uav,
sigma2_best), optionally conditioned on argmax(b_fused) == UAV (the frames the
detection gate would actually emit).

Key difference from the tracker script: b_bg is read from the log column
(present in the det-only *_peak fusion logs).  It is NOT derived as
1 - b_uav - u, because for naive/gated fusion u_fused is a pseudo-uncertainty
(1 - b_uav), so that derivation is wrong.  Only single-modality logs (which
lack a b_bg column and DO satisfy b_uav + b_bg + u = 1) fall back to the
derived value.

Usage:
    python scripts/analyse_uncertainty_separability_detonly.py \\
        --results_dir  .../val_results_detector_only_dbf_peak \\
        --dataset_root .../Anti-UAV-RGBT/val \\
        --modality     fusion \\
        --success_iou  0.5 \\
        --out_dir      reports/separability_detonly_dbf_val
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

U_COLS     = ("u_fused", "u")
SIGMA_COLS = ("sigma2_best", "sigma2")


# ---------------------------------------------------------------------------
# Loading
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
                preds.append([0.0, 0.0, 0.0, 0.0]); continue
            parts = line.replace("\t", ",").split(",")
            preds.append([float(p) for p in parts[:4]])
    return preds


def load_labels(seq_dir: Path):
    with open(seq_dir / "IR_label.json") as f:
        ir = json.load(f)
    with open(seq_dir / "visible_label.json") as f:
        rgb = json.load(f)
    return ir["exist"], ir["gt_rect"], rgb["exist"], rgb["gt_rect"]


def _detect_cols(rows):
    if not rows:
        return None, None
    keys = rows[0].keys()
    u_col     = next((c for c in U_COLS     if c in keys), None)
    sigma_col = next((c for c in SIGMA_COLS if c in keys), None)
    return u_col, sigma_col


def _f(v):
    if v in (None, "", "None"):
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Per-frame record building (stateless; optional argmax==UAV conditioning)
# ---------------------------------------------------------------------------

def build_records(results_dir: Path, dataset_root: Path, modality: str,
                  conditioned: bool):
    records = []
    n_present = n_excl_nobuav = n_excl_bg = 0

    for log_path in sorted(results_dir.glob("*_log.csv")):
        seq = log_path.name.replace("_log.csv", "")
        if seq.endswith("_ir"):          # skip auxiliary *_ir.txt companion logs
            continue
        pred_path = results_dir / f"{seq}.txt"
        seq_dir   = dataset_root / seq
        if not pred_path.exists() or not seq_dir.exists():
            continue

        rows = load_seq_log(log_path)
        u_col, sigma_col = _detect_cols(rows)
        if u_col is None or sigma_col is None:
            print(f"[warn] {log_path.name}: no u/sigma column, skipping")
            continue
        preds = load_predictions(pred_path)
        ir_exist, ir_gt, rgb_exist, rgb_gt = load_labels(seq_dir)
        has_bbg = "b_bg" in (rows[0].keys() if rows else [])

        n = min(len(rows), len(preds), len(ir_exist), len(rgb_exist))
        for i in range(n):
            r = rows[i]
            u = _f(r.get(u_col)); s = _f(r.get(sigma_col)); b_uav = _f(r.get("b_uav"))
            if u is None or s is None:
                continue

            present = bool(rgb_exist[i]) if modality == "rgb" else bool(ir_exist[i])
            if not present:
                continue
            n_present += 1

            # b_bg: read logged column (correct for naive/gated); else derive
            # (valid only for single-modality logs where b_uav+b_bg+u=1).
            b_bg = _f(r.get("b_bg")) if has_bbg else (
                None if b_uav is None else 1.0 - b_uav - u)

            if conditioned:
                if b_uav is None or b_bg is None:
                    n_excl_nobuav += 1; continue
                if b_uav <= b_bg:              # argmax(b_fused) == background
                    n_excl_bg += 1; continue

            # GT in the predicted box's coordinate space
            if modality == "ir":
                gt = ir_gt[i]
            elif modality == "rgb":
                gt = rgb_gt[i]
            else:
                winner = (r.get("dms_winner") or "ir")
                gt = rgb_gt[i] if (winner == "rgb" and bool(rgb_exist[i]) and rgb_gt[i] is not None) else ir_gt[i]
            if gt is None:
                continue

            pred = preds[i]
            iou = 0.0 if (pred[2] <= 0 or pred[3] <= 0) else iou_xywh(pred, gt)
            records.append({"u_fused": u, "b_uav": b_uav if b_uav is not None else float("nan"),
                            "sigma2_best": s, "iou": iou})

    print(f"  present frames seen:           {n_present}")
    if conditioned:
        print(f"  excluded — b_uav/b_bg missing: {n_excl_nobuav}")
        print(f"  excluded — argmax==background: {n_excl_bg}")
    print(f"  retained:                      {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_hist(out_path, succ, fail, xlabel, title, log_x=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if log_x:
        succ, fail = succ[succ > 0], fail[fail > 0]
        allv = np.concatenate([succ, fail])
        if allv.size == 0:
            plt.close(fig); return
        edges = np.linspace(np.log10(allv.min()), np.log10(allv.max()), 61)
        ax.hist(np.log10(succ), bins=edges, alpha=0.55, color="C2", label=f"success (n={succ.size})")
        ax.hist(np.log10(fail), bins=edges, alpha=0.55, color="C3", label=f"failure (n={fail.size})")
        ax.set_xlabel(f"log10({xlabel})")
    else:
        allv = np.concatenate([succ, fail])
        if allv.size == 0:
            plt.close(fig); return
        edges = np.linspace(float(allv.min()), float(allv.max()), 61)
        ax.hist(succ, bins=edges, alpha=0.55, color="C2", label=f"success (n={succ.size})")
        ax.hist(fail, bins=edges, alpha=0.55, color="C3", label=f"failure (n={fail.size})")
        ax.set_xlabel(xlabel)
    ax.set_ylabel("count"); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_roc(out_path, signals, failures, title):
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        return
    if len(np.unique(failures)) < 2:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for label, scores in signals.items():
        fpr, tpr, _ = roc_curve(failures, scores)
        ax.plot(fpr, tpr, label=f"{label} (AUROC={compute_auroc(scores, failures):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FPR (good frames flagged)"); ax.set_ylabel("TPR (failures caught)")
    ax.set_title(title); ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",  required=True, type=Path)
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--out_dir",      required=True, type=Path)
    ap.add_argument("--modality", choices=("fusion", "ir", "rgb"), default="fusion")
    ap.add_argument("--success_iou", type=float, default=0.5)
    ap.add_argument("--unconditioned", action="store_true",
                    help="Analyse all present frames, not just argmax(b_fused)==UAV.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    conditioned = not args.unconditioned

    print(f"Results dir : {args.results_dir}")
    print(f"Modality    : {args.modality}   conditioned(argmax==UAV): {conditioned}")
    records = build_records(args.results_dir, args.dataset_root, args.modality, conditioned)
    if len(records) < 10:
        print("Not enough frames; aborting."); return

    u   = np.array([r["u_fused"] for r in records], float)
    b   = np.array([r["b_uav"] for r in records], float)
    sig = np.array([r["sigma2_best"] for r in records], float)
    iou = np.array([r["iou"] for r in records], float)
    failure = (iou < args.success_iou).astype(int)

    n_fail = int(failure.sum()); n_succ = int(len(failure) - n_fail)
    print(f"  success={n_succ}  failure={n_fail}  mean IoU={iou.mean():.4f}  "
          f"success rate(IoU>={args.success_iou})={np.mean(iou >= args.success_iou):.4f}")

    out = {
        "results_dir": str(args.results_dir),
        "modality": args.modality,
        "conditioned_on": "argmax(b_fused)==UAV" if conditioned else "all present frames",
        "success_iou": args.success_iou,
        "n_frames": len(records), "n_success": n_succ, "n_failure": n_fail,
        "mean_iou": float(iou.mean()),
        "signals": {},
    }

    if n_succ < 5 or n_fail < 5:
        print("  [skip] not enough of both classes for AUROC")
    else:
        # b_uav negated → higher belief predicts SUCCESS (lower failure),
        # matching the tracker separability table's convention.
        triples = [("u_fused", u, False), ("b_uav", b, True), ("sigma2_best", sig, False)]
        roc_signals = {}
        for name, vals, negate in triples:
            score = -vals if negate else vals
            auc = compute_auroc(score, failure)
            out["signals"][name] = {
                "auroc_failure": float(auc),
                "mean_success": float(vals[failure == 0].mean()),
                "mean_failure": float(vals[failure == 1].mean()),
                "median_success": float(np.median(vals[failure == 0])),
                "median_failure": float(np.median(vals[failure == 1])),
                "negated_for_auroc": negate,
            }
            roc_signals["b_uav (neg)" if negate else name] = score
            plot_hist(args.out_dir / f"hist_{name}.png",
                      vals[failure == 0], vals[failure == 1], name,
                      f"{args.modality} | {name}  (AUROC={auc:.3f}, IoU>={args.success_iou})",
                      log_x=(name == "sigma2_best"))
            print(f"  {name:13s} AUROC(failure)={auc:.3f}  "
                  f"mean succ={vals[failure==0].mean():.4g}  fail={vals[failure==1].mean():.4g}")
        plot_roc(args.out_dir / "roc.png", roc_signals, failure,
                 f"{args.modality} det-only — failure detection ROC (IoU>={args.success_iou})")

    with open(args.out_dir / "separability_detonly.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out_dir / 'separability_detonly.json'}")


if __name__ == "__main__":
    main()
