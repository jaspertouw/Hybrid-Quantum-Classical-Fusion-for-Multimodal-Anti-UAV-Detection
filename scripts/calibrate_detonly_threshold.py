"""
calibrate_detonly_threshold.py
==============================
EXACT offline sweep of the detector-only uncertainty threshold (theta_cls).

For the detector-only fusion variants (dbf / abf / naive / gated) the only
decision the gate makes per frame is:

    emit DMS box   iff   b_uav > b_bg  AND  u_fused < theta_cls
    else            ->   [0, 0, 0, 0]

The DMS box, dms_winner, b_uav, b_bg and u_fused are all logged per frame
(pre-gate) by run_detector_only.py.  None of them depend on theta_cls, so
recomputing Acc for any theta from the logs is EXACT — no GPU inference and no
approximation (unlike the tracker's calibrate_thresholds_offline.py, which must
replay a state machine).

Workflow
--------
  1. Run the variant once on val (theta_cls=1.0 → argmax-only headline Acc):
        sbatch scripts/eval_detector_only_dbf.job        # etc.
  2. Sweep theta over the produced logs (this script, seconds on CPU):
        python scripts/calibrate_detonly_threshold.py \\
            --log_dir      .../val_results_detector_only_dbf \\
            --dataset_root .../Anti-UAV-RGBT/val \\
            --out_dir      .../calibration_detonly_dbf
  3. Re-run the variant on TEST with --theta_cls <best> (test job script).

Validation: theta_cls=1.0 in the grid should reproduce the run's summary.json
Acc to within rounding (it applies the identical argmax-only gate).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os


# ---------------------------------------------------------------------------
# EDTC Acc (DMS-aware) — inlined from run_edtc_fusion.py (no GPU/ML deps)
# ---------------------------------------------------------------------------

def iou_xywh(pred, gt):
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px + pw, gx + gw), min(py + ph, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return inter / (union + 1e-8)


def compute_edtc_acc_dms(preds, frames, dms_winners,
                         alpha=0.2, beta=0.3, failure_iou_thresh=1e-5):
    """Option-B DMS-aware Acc: existence anchored to IR GT; IoU vs winner GT."""
    T = len(frames)
    if T == 0:
        return float("nan")
    reward_sum, failure_sum = 0.0, 0.0
    for pred, frame, winner in zip(preds, frames, dms_winners):
        ir_gt  = frame["gt"]
        rgb_gt = frame.get("rgb_gt")
        pred_empty = (pred is None or pred[2] <= 0 or pred[3] <= 0)
        p_t = 1.0 if pred_empty else 0.0
        if ir_gt is not None:
            gt_for_iou = rgb_gt if winner == "rgb" else ir_gt
            iou_t = iou_xywh(pred, gt_for_iou) if (gt_for_iou is not None and not pred_empty) else 0.0
            reward_sum += iou_t
            if iou_t < failure_iou_thresh:
                failure_sum += 1.0
        else:
            reward_sum += p_t
    first_term = reward_sum / T
    penalty    = alpha * ((failure_sum / T) ** beta) if T > 0 else 0.0
    return first_term - penalty


# ---------------------------------------------------------------------------
# Ground-truth loader (bbox/existence only — no images)
# ---------------------------------------------------------------------------

def load_gt(seq_dir: str):
    """Return list of {gt: [x,y,w,h]|None, rgb_gt: [x,y,w,h]|None} per frame."""
    with open(os.path.join(seq_dir, "IR_label.json")) as f:
        ir_ann = json.load(f)
    with open(os.path.join(seq_dir, "visible_label.json")) as f:
        rgb_ann = json.load(f)
    n = min(len(ir_ann["exist"]), len(rgb_ann["exist"]))
    return [{
        "gt":     ir_ann["gt_rect"][i]  if ir_ann["exist"][i]  else None,
        "rgb_gt": rgb_ann["gt_rect"][i] if rgb_ann["exist"][i] else None,
    } for i in range(n)]


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _f(x):
    return float(x) if x not in (None, "") else None


def parse_log(log_path: str):
    """Parse one <seq>_log.csv into a list of per-frame dicts."""
    rows = []
    with open(log_path, newline="") as f:
        for r in csv.DictReader(f):
            dms_raw = r.get("dms_box") or ""
            rows.append({
                "u_fused":    _f(r.get("u_fused")),
                "b_uav":      _f(r.get("b_uav")),
                "b_bg":       _f(r.get("b_bg")),
                "dms_box":    json.loads(dms_raw) if dms_raw else None,
                "dms_winner": (r.get("dms_winner") or "") or None,
            })
    return rows


def build_preds(rows, theta_cls):
    """Apply the gate at a given theta_cls.  Returns (preds, dms_winners)."""
    preds, winners = [], []
    for r in rows:
        box = r["dms_box"]
        if box is None or r["b_uav"] is None:
            preds.append([0.0, 0.0, 0.0, 0.0])
        elif r["b_uav"] > r["b_bg"] and r["u_fused"] < theta_cls:
            preds.append(box)
        else:
            preds.append([0.0, 0.0, 0.0, 0.0])
        winners.append(r["dms_winner"] or "ir")
    return preds, winners


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Exact offline det-only theta_cls sweep")
    p.add_argument("--log_dir",      required=True, help="dir with <seq>_log.csv from a det-only run")
    p.add_argument("--dataset_root", required=True, help="split root (val or test) for GT")
    p.add_argument("--out_dir",      default=None,  help="where to write sweep results (default: log_dir)")
    p.add_argument("--theta_grid",   type=float, nargs="*", default=None,
                   help="theta_cls values to sweep (default: 0.05..0.95 step 0.05)")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta",  type=float, default=0.3)
    p.add_argument("--failure_iou_thresh", type=float, default=1e-5)
    p.add_argument("--top_k", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or args.log_dir
    os.makedirs(out_dir, exist_ok=True)

    grid = args.theta_grid or [round(0.05 * k, 2) for k in range(1, 20)]  # 0.05..0.95

    log_files = sorted(glob.glob(os.path.join(args.log_dir, "*_log.csv")))
    if not log_files:
        raise SystemExit(f"No *_log.csv found in {args.log_dir}")

    # Parse each sequence's log + GT ONCE; sweep thetas in memory.
    cached = []   # list of (seq_name, rows, gt_frames)
    for lf in log_files:
        seq_name = os.path.basename(lf)[:-len("_log.csv")]
        seq_dir  = os.path.join(args.dataset_root, seq_name)
        if not os.path.isdir(seq_dir):
            print(f"[skip] no GT dir for {seq_name}")
            continue
        rows = parse_log(lf)
        gt   = load_gt(seq_dir)
        n = min(len(rows), len(gt))
        if len(rows) != len(gt):
            print(f"[warn] {seq_name}: {len(rows)} log rows vs {len(gt)} GT frames -> using {n}")
        cached.append((seq_name, rows[:n], gt[:n]))

    # u_fused distribution on gate-eligible frames (dms_box present AND
    # b_uav > b_bg) — the only frames where theta_cls can change the output.
    # If max(u) < min(grid), every theta in the grid acts identically and the
    # gate reduces to argmax(b_fused).
    u_vals = sorted(
        r["u_fused"]
        for _seq, rows, _gt in cached for r in rows
        if r["dms_box"] is not None and r["b_uav"] is not None
        and r["b_bg"] is not None and r["u_fused"] is not None
        and r["b_uav"] > r["b_bg"]
    )
    u_stats = None
    if u_vals:
        q = lambda p: u_vals[min(len(u_vals) - 1, int(p * len(u_vals)))]
        u_stats = {
            "n_gate_eligible": len(u_vals),
            "min": u_vals[0], "p50": q(0.50), "p99": q(0.99), "max": u_vals[-1],
            "frac_above_grid_min": sum(v >= grid[0] for v in u_vals) / len(u_vals),
            "grid_min": grid[0],
        }
        print(f"u_fused on gate-eligible frames (n={len(u_vals)}):")
        print(f"  min={u_vals[0]:.4g}  p50={q(0.50):.4g}  p99={q(0.99):.4g}  max={u_vals[-1]:.4g}")
        print(f"  fraction >= grid min ({grid[0]}): {u_stats['frac_above_grid_min']:.6f}")
        if u_vals[-1] < grid[0]:
            print(f"  -> theta_cls is INERT over this grid: all u_fused < {grid[0]}")
        print()

    print(f"Loaded {len(cached)} sequences.  Sweeping {len(grid)} theta_cls values...\n")

    results = []
    for theta in grid:
        accs = []
        for _seq, rows, gt in cached:
            preds, winners = build_preds(rows, theta)
            acc = compute_edtc_acc_dms(preds, gt, winners,
                                       alpha=args.alpha, beta=args.beta,
                                       failure_iou_thresh=args.failure_iou_thresh)
            if not math.isnan(acc):
                accs.append(acc)
        mean_acc = float(sum(accs) / len(accs)) if accs else float("nan")
        results.append({"theta_cls": theta, "acc": mean_acc, "n_sequences": len(accs)})
        print(f"  theta_cls={theta:.3f}   Acc={mean_acc:.4f}   [{len(accs)} seqs]")

    ranked = sorted((r for r in results if not math.isnan(r["acc"])),
                    key=lambda r: r["acc"], reverse=True)
    best = ranked[0] if ranked else None

    print("\n" + "=" * 60)
    if best:
        print(f"BEST  theta_cls={best['theta_cls']:.3f}   Acc={best['acc']:.4f}")
    print("Top candidates:")
    for r in ranked[:args.top_k]:
        print(f"  theta_cls={r['theta_cls']:.3f}   Acc={r['acc']:.4f}")
    print("=" * 60)

    accs_seen = {round(r["acc"], 6) for r in results if not math.isnan(r["acc"])}
    theta_inert = len(accs_seen) <= 1
    if theta_inert:
        print("\nNOTE: Acc is identical for every theta in the grid -> theta_cls inert;"
              " gate reduces to argmax(b_fused).")

    out = {
        "log_dir":      args.log_dir,
        "u_fused_gate_eligible_stats": u_stats,
        "theta_inert_over_grid": theta_inert,
        "dataset_root": args.dataset_root,
        "grid":         grid,
        "results":      results,
        "ranked":       ranked,
        "best":         best,
        "alpha":        args.alpha,
        "beta":         args.beta,
        "failure_iou_thresh": args.failure_iou_thresh,
    }
    out_path = os.path.join(out_dir, "theta_sweep.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSweep results -> {out_path}")


if __name__ == "__main__":
    main()
