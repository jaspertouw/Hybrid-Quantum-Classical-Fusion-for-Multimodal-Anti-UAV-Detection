"""
analyse_uncertainty_separability.py
===================================
Offline separability analysis for the EDTC switching signals (u_fused, sigma2_best).

Joins each <seq>_log.csv with its prediction <seq>.txt and the dataset's GT
label JSONs to obtain a per-frame IoU. Labels frames as success (IoU >= τ) or
failure, then for each tracker state (DETECT, TRACK) plots overlapping
success/failure histograms of u_fused and sigma2_best and computes AUROC
treating uncertainty as a failure score.

If the current run's summary.json is present, the thresholds it used are also
evaluated: TPR (fraction of failures caught) and FPR (fraction of successes
wrongly rejected) at those thresholds.

Usage:
    python scripts/analyse_uncertainty_separability.py \
        --results_dir /path/to/val_results_dbf \
        --dataset_root /path/to/Anti-UAV-RGBT/val \
        --success_iou 0.5 \
        --out_dir reports/separability_dbf_val
"""

from __future__ import annotations

import argparse
import csv
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

from resources.EDTC.lib.utils.eval_metrics import compute_auroc, iou_xywh


STATES = ("DETECT", "TRACK")
# Column name aliases: fusion logs use u_fused/sigma2_best; single-modality uses u/sigma2
U_COLS     = ("u_fused", "u")
SIGMA_COLS = ("sigma2_best", "sigma2")
THRESH_KEYS = {
    ("DETECT", "u_fused"):     "theta_cls_detect",
    ("DETECT", "u"):           "theta_cls_detect",
    ("DETECT", "sigma2_best"): "theta_loc_detect",
    ("DETECT", "sigma2"):      "theta_loc_detect",
    ("TRACK",  "u_fused"):     "theta_cls_track",
    ("TRACK",  "u"):           "theta_cls_track",
    ("TRACK",  "sigma2_best"): "theta_loc_track",
    ("TRACK",  "sigma2"):      "theta_loc_track",
    # b_uav has no threshold key yet (proposed replacement signal)
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_seq_log(csv_path: Path):
    """Return list of per-frame dicts from <seq>_log.csv."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_predictions(txt_path: Path):
    """Predicted boxes in DMS-winner coordinate space — one [x,y,w,h] per line."""
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
    """Return (ir_exist, ir_gt, rgb_exist, rgb_gt) lists."""
    with open(seq_dir / "IR_label.json") as f:
        ir = json.load(f)
    with open(seq_dir / "visible_label.json") as f:
        rgb = json.load(f)
    return ir["exist"], ir["gt_rect"], rgb["exist"], rgb["gt_rect"]


# ---------------------------------------------------------------------------
# Per-frame record building
# ---------------------------------------------------------------------------

def _detect_u_sigma_cols(rows):
    """Return (u_col, sigma_col) from the first row's keys."""
    if not rows:
        return None, None
    keys = rows[0].keys()
    u_col     = next((c for c in U_COLS     if c in keys), None)
    sigma_col = next((c for c in SIGMA_COLS if c in keys), None)
    return u_col, sigma_col


def build_records(results_dir: Path, dataset_root: Path, modality: str = "fusion"):
    """Yield dicts {state, u_fused, b_uav, sigma2_best, iou} for every present frame.

    Column names are auto-detected: fusion logs use u_fused/sigma2_best,
    single-modality logs use u/sigma2. Internally always stored as u_fused/sigma2_best.
    Frames where target is absent are excluded (success is ill-defined there).

    modality:
      "fusion" — presence = IR-exist, GT picked by dms_winner (fusion behavior).
      "ir"     — presence = IR-exist, GT = IR boxes.
      "rgb"    — presence = RGB-exist, GT = RGB boxes.
    """
    assert modality in ("fusion", "ir", "rgb"), modality
    records = []
    missing_seqs = []
    for log_path in sorted(results_dir.glob("*_log.csv")):
        seq_name = log_path.name.replace("_log.csv", "")
        if seq_name.endswith("_ir"):
            continue  # ignore IR-only side outputs if any
        pred_path = results_dir / f"{seq_name}.txt"
        seq_dir = dataset_root / seq_name
        if not pred_path.exists() or not seq_dir.exists():
            missing_seqs.append(seq_name)
            continue

        rows = load_seq_log(log_path)
        u_col, sigma_col = _detect_u_sigma_cols(rows)
        if u_col is None or sigma_col is None:
            print(f"[warn] {log_path.name}: cannot detect u/sigma columns, skipping")
            continue
        preds = load_predictions(pred_path)
        ir_exist, ir_gt, rgb_exist, rgb_gt = load_labels(seq_dir)

        n = min(len(rows), len(preds), len(ir_exist), len(rgb_exist))
        for i in range(n):
            r = rows[i]
            u_raw = r.get(u_col, "")
            s_raw = r.get(sigma_col, "")
            if u_raw == "" or s_raw == "":
                continue
            try:
                u = float(u_raw)
                s = float(s_raw)
            except ValueError:
                continue
            if not (math.isfinite(u) and math.isfinite(s)):
                continue

            b_raw = r.get("b_uav", "")
            try:
                b_uav = float(b_raw) if b_raw != "" else None
            except ValueError:
                b_uav = None

            state = r.get("state", "")
            if state not in STATES:
                continue

            if modality == "rgb":
                present = bool(rgb_exist[i])
            else:
                present = bool(ir_exist[i])
            if not present:
                continue

            if modality == "ir":
                gt_box = ir_gt[i]
            elif modality == "rgb":
                gt_box = rgb_gt[i]
            else:
                winner = r.get("dms_winner", "ir") or "ir"
                if winner == "rgb" and bool(rgb_exist[i]) and rgb_gt[i] is not None:
                    gt_box = rgb_gt[i]
                else:
                    gt_box = ir_gt[i]
            if gt_box is None:
                continue

            pred = preds[i]
            if pred[2] <= 0 or pred[3] <= 0:
                iou = 0.0
            else:
                iou = iou_xywh(pred, gt_box)

            records.append({
                "seq": seq_name,
                "frame": i,
                "state": state,
                "u_fused": u,
                "b_uav": b_uav,
                "sigma2_best": s,
                "iou": iou,
            })

    if missing_seqs:
        print(f"[warn] {len(missing_seqs)} sequences missing predictions or label dir; skipped")
    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def auroc_failure_score(scores: np.ndarray, failures: np.ndarray) -> float:
    """AUROC with the score predicting failure (higher score = more likely fail).

    `compute_auroc(confs, correctness)` expects "higher confs ⇒ correct ⇒ label 1".
    Pass `confs=scores`, `correctness=failures` to get AUROC for failure detection.
    """
    return compute_auroc(scores, failures)


def threshold_op_point(scores: np.ndarray, failures: np.ndarray, thresh: float):
    """At decision rule (score >= thresh ⇒ flag as failure), return (TPR, FPR, PPV)."""
    flag = scores >= thresh
    fail = failures.astype(bool)
    tp = int(np.sum(flag & fail))
    fp = int(np.sum(flag & ~fail))
    fn = int(np.sum(~flag & fail))
    tn = int(np.sum(~flag & ~fail))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {"threshold": float(thresh), "tpr": tpr, "fpr": fpr, "ppv": ppv,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def plot_hist(out_path: Path, succ: np.ndarray, fail: np.ndarray,
              xlabel: str, title: str, log_x: bool = False, thresh: float | None = None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = 60
    if log_x:
        succ_p = succ[succ > 0]
        fail_p = fail[fail > 0]
        all_p = np.concatenate([succ_p, fail_p])
        if all_p.size == 0:
            plt.close(fig)
            return
        lo, hi = np.log10(all_p.min()), np.log10(all_p.max())
        bin_edges = np.linspace(lo, hi, bins + 1)
        ax.hist(np.log10(succ_p), bins=bin_edges, alpha=0.55, color="C2",
                label=f"success (n={succ.size})")
        ax.hist(np.log10(fail_p), bins=bin_edges, alpha=0.55, color="C3",
                label=f"failure (n={fail.size})")
        ax.set_xlabel(f"log10({xlabel})")
        if thresh is not None and thresh > 0:
            ax.axvline(np.log10(thresh), color="k", ls="--", lw=1.0,
                       label=f"threshold = {thresh:.3g}")
    else:
        all_v = np.concatenate([succ, fail])
        if all_v.size == 0:
            plt.close(fig)
            return
        lo, hi = float(all_v.min()), float(all_v.max())
        bin_edges = np.linspace(lo, hi, bins + 1)
        ax.hist(succ, bins=bin_edges, alpha=0.55, color="C2",
                label=f"success (n={succ.size})")
        ax.hist(fail, bins=bin_edges, alpha=0.55, color="C3",
                label=f"failure (n={fail.size})")
        ax.set_xlabel(xlabel)
        if thresh is not None:
            ax.axvline(thresh, color="k", ls="--", lw=1.0,
                       label=f"threshold = {thresh:.3g}")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_roc(out_path: Path, signals: dict, failures: np.ndarray, title: str):
    """signals: {label: scores}; one ROC per signal."""
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for label, scores in signals.items():
        if len(np.unique(failures)) < 2:
            continue
        fpr, tpr, _ = roc_curve(failures, scores)
        auc = compute_auroc(scores, failures)
        ax.plot(fpr, tpr, label=f"{label} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FPR (good frames wrongly flagged)")
    ax.set_ylabel("TPR (failures caught)")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def shuffle_sanity_check(scores: np.ndarray, failures: np.ndarray, n_trials: int = 20):
    """Random shuffles of the failure labels should give AUROC ≈ 0.5."""
    rng = np.random.default_rng(0)
    aurocs = []
    for _ in range(n_trials):
        shuffled = failures.copy()
        rng.shuffle(shuffled)
        aurocs.append(auroc_failure_score(scores, shuffled))
    mean = float(np.mean(aurocs))
    assert 0.4 < mean < 0.6, f"Shuffle sanity failed: mean AUROC={mean:.3f}"
    return mean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, type=Path,
                    help="Directory with *_log.csv, *.txt, summary.json")
    ap.add_argument("--dataset_root", required=True, type=Path,
                    help="e.g. Anti-UAV-RGBT/val containing per-seq dirs")
    ap.add_argument("--success_iou", type=float, default=0.5)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--modality", choices=("fusion", "ir", "rgb"), default="fusion",
                    help="fusion: dms_winner-driven GT (default). ir/rgb: single-modality GT and presence flag.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: thresholds used at evaluation time
    summary_path = args.results_dir / "summary.json"
    run_thresholds = None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        run_thresholds = summary.get("thresholds")
        print(f"Loaded thresholds from summary.json: {run_thresholds}")

    print("Building per-frame records...")
    records = build_records(args.results_dir, args.dataset_root, modality=args.modality)
    print(f"  total present frames: {len(records)}")
    if not records:
        print("No usable frames. Aborting.")
        return

    arr_state  = np.array([r["state"] for r in records])
    arr_u      = np.array([r["u_fused"] for r in records], dtype=np.float64)
    arr_sigma  = np.array([r["sigma2_best"] for r in records], dtype=np.float64)
    arr_iou    = np.array([r["iou"] for r in records], dtype=np.float64)
    failure    = (arr_iou < args.success_iou).astype(np.int32)
    has_b_uav  = any(r["b_uav"] is not None for r in records)
    arr_b_uav  = np.array([r["b_uav"] if r["b_uav"] is not None else float("nan")
                            for r in records], dtype=np.float64) if has_b_uav else None
    if not has_b_uav:
        print("  [info] b_uav column not present in logs — re-run inference to populate it")

    overall_succ_rate = float(np.mean(arr_iou >= args.success_iou))
    print(f"  overall success rate (IoU>={args.success_iou}): {overall_succ_rate:.4f}")

    result = {
        "n_frames_total": int(len(records)),
        "success_iou": args.success_iou,
        "overall_success_rate": overall_succ_rate,
        "run_thresholds": run_thresholds,
        "per_state": {},
    }

    for state in STATES:
        mask = arr_state == state
        n = int(mask.sum())
        u_s    = arr_u[mask]
        sig_s  = arr_sigma[mask]
        fail_s = failure[mask]
        iou_s  = arr_iou[mask]
        b_s    = arr_b_uav[mask] if arr_b_uav is not None else None
        n_fail = int(fail_s.sum())
        n_succ = int(n - n_fail)
        print(f"\n[{state}] n={n}  success={n_succ}  failure={n_fail}")

        state_block = {"n_total": n, "n_success": n_succ, "n_failure": n_fail,
                       "mean_iou": float(iou_s.mean()) if n > 0 else float("nan"),
                       "signals": {}}

        if n_succ < 5 or n_fail < 5:
            print("  [skip] not enough of both classes to compute AUROC")
            result["per_state"][state] = state_block
            continue

        signal_triples = [("u_fused", u_s, fail_s), ("sigma2_best", sig_s, fail_s)]
        if b_s is not None:
            valid = ~np.isnan(b_s)
            if valid.sum() > 10:
                signal_triples.insert(1, ("b_uav", b_s[valid], fail_s[valid]))

        for signal_name, signal_vals, fail_s_sig in signal_triples:
            auc = auroc_failure_score(signal_vals, fail_s_sig)
            mean_succ = float(signal_vals[fail_s_sig == 0].mean())
            mean_fail = float(signal_vals[fail_s_sig == 1].mean())
            med_succ  = float(np.median(signal_vals[fail_s_sig == 0]))
            med_fail  = float(np.median(signal_vals[fail_s_sig == 1]))

            sig_block = {
                "auroc_failure": auc,
                "mean_success": mean_succ,
                "mean_failure": mean_fail,
                "median_success": med_succ,
                "median_failure": med_fail,
            }

            # operating point at the run's threshold, if available
            thresh_val = None
            if run_thresholds is not None:
                key = THRESH_KEYS.get((state, signal_name))
                if key and key in run_thresholds:
                    thresh_val = float(run_thresholds[key])
                    sig_block["operating_point"] = threshold_op_point(
                        signal_vals, fail_s_sig, thresh_val)

            # shuffle sanity check
            try:
                sig_block["shuffle_auroc_mean"] = shuffle_sanity_check(signal_vals, fail_s_sig)
            except AssertionError as e:
                sig_block["shuffle_auroc_mean"] = f"FAIL: {e}"

            state_block["signals"][signal_name] = sig_block

            # plot histograms
            succ_v = signal_vals[fail_s_sig == 0]
            fail_v = signal_vals[fail_s_sig == 1]
            log_x = (signal_name == "sigma2_best")
            plot_hist(
                args.out_dir / f"hist_{signal_name}_{state}.png",
                succ_v, fail_v,
                xlabel=signal_name,
                title=f"{state} — {signal_name}  (AUROC={auc:.3f}, IoU≥{args.success_iou})",
                log_x=log_x,
                thresh=thresh_val,
            )
            print(f"  {signal_name:13s}  AUROC={auc:.3f}  "
                  f"mean(succ)={mean_succ:.4f}  mean(fail)={mean_fail:.4f}"
                  + (f"  thresh={thresh_val:.4g}" if thresh_val is not None else ""))

        # combined ROC plot for this state (include b_uav if available)
        roc_signals = {"u_fused": u_s, "sigma2_best": sig_s}
        if b_s is not None:
            valid = ~np.isnan(b_s)
            if valid.sum() > 10:
                roc_signals["b_uav"] = b_s[valid]
        plot_roc(
            args.out_dir / f"roc_{state}.png",
            roc_signals,
            fail_s,
            title=f"{state} — failure detection ROC (IoU≥{args.success_iou})",
        )

        result["per_state"][state] = state_block

    with open(args.out_dir / "separability.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out_dir / 'separability.json'}")

    write_report(args.out_dir / "separability_report.md", result)
    print(f"Wrote {args.out_dir / 'separability_report.md'}")


def write_report(path: Path, result: dict):
    lines = []
    lines.append("# Uncertainty Separability Report\n")
    lines.append(f"- Success criterion: IoU ≥ **{result['success_iou']}**")
    lines.append(f"- Total present frames analysed: **{result['n_frames_total']}**")
    lines.append(f"- Overall success rate: **{result['overall_success_rate']:.4f}**")
    if result.get("run_thresholds"):
        lines.append(f"- Thresholds used by the evaluated run: `{result['run_thresholds']}`")
    lines.append("")

    for state, block in result["per_state"].items():
        lines.append(f"## {state}")
        lines.append(f"- frames: {block['n_total']} (success={block['n_success']}, "
                     f"failure={block['n_failure']}, mean IoU={block.get('mean_iou', float('nan')):.3f})")
        if not block.get("signals"):
            lines.append("- _Insufficient data for AUROC._\n")
            continue
        for sig_name, sig in block["signals"].items():
            auc = sig["auroc_failure"]
            verdict = ("strong" if auc >= 0.8 else
                       "moderate" if auc >= 0.7 else
                       "weak" if auc >= 0.6 else "no")
            line = (f"- **{sig_name}** — AUROC(failure) = **{auc:.3f}** "
                    f"({verdict} separability). "
                    f"mean: succ={sig['mean_success']:.4g}, fail={sig['mean_failure']:.4g}; "
                    f"median: succ={sig['median_success']:.4g}, fail={sig['median_failure']:.4g}.")
            lines.append(line)
            op = sig.get("operating_point")
            if op:
                lines.append(
                    f"  - At run threshold = {op['threshold']:.4g}: "
                    f"TPR (failures flagged) = {op['tpr']:.3f}, "
                    f"FPR (good frames flagged) = {op['fpr']:.3f}, "
                    f"PPV = {op['ppv']:.3f}.")
            shuf = sig.get("shuffle_auroc_mean")
            if isinstance(shuf, float):
                lines.append(f"  - Shuffle-sanity AUROC ≈ {shuf:.3f} (expected ≈ 0.5).")
        lines.append("")

    lines.append("## Interpretation guide")
    lines.append("- AUROC ≥ 0.8 → signal cleanly separates success/failure; p95 threshold "
                 "choice has empirical footing.")
    lines.append("- 0.6 ≤ AUROC < 0.8 → useful but noisy; thresholds should be tuned, not "
                 "picked by percentile alone.")
    lines.append("- AUROC < 0.6 → distributions overlap heavily; the threshold-based "
                 "switching logic is essentially gating on noise, regardless of percentile. "
                 "The calibration approach itself needs to be revisited.")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
