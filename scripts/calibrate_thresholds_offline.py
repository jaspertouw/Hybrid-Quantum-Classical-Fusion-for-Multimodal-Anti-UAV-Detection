"""
calibrate_thresholds_offline.py
================================
Offline threshold calibration for the EDTC fusion (or single-modality) tracker.

Instead of running full GPU inference for every threshold candidate, this script:
  1. Loads per-frame *_log.csv files already produced by a previous run.
  2. Replays the two-state switching machine (DETECT / TRACK) with new threshold
     4-tuples using the stored uncertainty signals (u_fused, sigma2_best) and the
     stored per-modality bboxes as predictions.
  3. Evaluates EDTC Acc by loading ground-truth from the dataset.
  4. Sweeps a user-supplied threshold grid and ranks by Acc.

APPROXIMATION WARNING
---------------------
The stored u/sigma2 values come from whichever model was active at that frame in
the ORIGINAL run (YOLO in DETECT, SiamCAR in TRACK).  When new thresholds shift a
state boundary, the replay uses the wrong model's signals for frames near the
transition.  The approximation is reliable when threshold changes are small (|Δ| <
~0.15).  The script reports a per-combo "divergence_rate" (fraction of frames
where the replayed state differs from the original) as a quality indicator; combos
with divergence_rate > 0.25 should be confirmed with a real GPU run.

USAGE EXAMPLES
--------------
# Phase A: inspect distributions first:
#   python scripts/plot_edtc_threshold_distributions.py ...
#
# Pass 1 — sweep DETECT pair, fix TRACK at baseline:
python scripts/calibrate_thresholds_offline.py \\
    --log_dir      /path/to/val_results_dbf \\
    --dataset_root /path/to/Anti-UAV-RGBT/val \\
    --theta_cls_detect 0.70 0.75 0.80 0.85 0.90 \\
    --theta_loc_detect 0.04 0.06 0.08 0.10 0.12 \\
    --theta_cls_track  0.90 \\
    --theta_loc_track  0.70 \\
    --out_dir /path/to/calibration_results

# Pass 2 — fix DETECT at best found, sweep TRACK pair:
python scripts/calibrate_thresholds_offline.py \\
    --log_dir      /path/to/val_results_dbf \\
    --dataset_root /path/to/Anti-UAV-RGBT/val \\
    --theta_cls_detect 0.80 \\
    --theta_loc_detect 0.08 \\
    --theta_cls_track  0.80 0.85 0.90 0.95 \\
    --theta_loc_track  0.60 0.65 0.70 0.75 0.80 \\
    --out_dir /path/to/calibration_results

# Validate baseline (should reproduce summary.json Acc within ~0.02):
python scripts/calibrate_thresholds_offline.py \\
    --log_dir      /path/to/val_results_dbf \\
    --dataset_root /path/to/Anti-UAV-RGBT/val \\
    --theta_cls_detect 0.80 \\
    --theta_loc_detect 0.10 \\
    --theta_cls_track  0.90 \\
    --theta_loc_track  0.70
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# GT loader (no image reading — only bbox/existence arrays needed)
# ---------------------------------------------------------------------------

def load_gt(seq_dir: str):
    """Return list of {gt: [x,y,w,h]|None, rgb_gt: [x,y,w,h]|None} dicts."""
    ir_label_path  = os.path.join(seq_dir, "IR_label.json")
    rgb_label_path = os.path.join(seq_dir, "visible_label.json")

    with open(ir_label_path)  as f: ir_ann  = json.load(f)
    with open(rgb_label_path) as f: rgb_ann = json.load(f)

    n = min(len(ir_ann["exist"]), len(rgb_ann["exist"]))
    frames = []
    for i in range(n):
        frames.append({
            "gt":     ir_ann["gt_rect"][i]  if ir_ann["exist"][i]  else None,
            "rgb_gt": rgb_ann["gt_rect"][i] if rgb_ann["exist"][i] else None,
        })
    return frames


# ---------------------------------------------------------------------------
# Log loader
# ---------------------------------------------------------------------------

def load_log(log_path: Path):
    """Return list of per-frame dicts from a *_log.csv file."""
    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u_raw = row.get("u_fused") or row.get("u") or ""
            s_raw = row.get("sigma2_best") or row.get("sigma2") or ""

            u      = float(u_raw) if u_raw else None
            sigma2 = float(s_raw) if s_raw else None

            # Parse stored bboxes
            ir_bbox  = _parse_bbox(row.get("ir_bbox",  "") or row.get("bbox", ""))
            rgb_bbox = _parse_bbox(row.get("rgb_bbox", ""))
            dms_win  = (row.get("dms_winner") or "ir").strip()
            orig_state = (row.get("state") or "DETECT").strip()

            rows.append({
                "u":          u,
                "sigma2":     sigma2,
                "ir_bbox":    ir_bbox,
                "rgb_bbox":   rgb_bbox,
                "dms_winner": dms_win,
                "orig_state": orig_state,
            })
    return rows


def _parse_bbox(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        vals = json.loads(raw)
        if isinstance(vals, list) and len(vals) == 4:
            return [float(v) for v in vals]
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        parts = [float(p) for p in raw.strip("[]").split(",")]
        if len(parts) == 4:
            return parts
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Metrics (copied from run_edtc_fusion.py — no GPU dependencies)
# ---------------------------------------------------------------------------

def iou_xywh(pred, gt):
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px + pw, gx + gw), min(py + ph, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return inter / (union + 1e-8)


def compute_edtc_acc(preds, frames, dms_winners=None,
                     alpha=0.2, beta=0.3, failure_iou_thresh=1e-5):
    """
    If dms_winners is provided, compute DMS-aware Acc (each frame scored against
    the winner modality's GT).  Otherwise, score all predictions against IR GT.
    """
    T = len(frames)
    if T == 0:
        return float("nan")

    reward_sum, failure_sum = 0.0, 0.0
    for i, (pred, frame) in enumerate(zip(preds, frames)):
        if dms_winners is not None:
            winner = dms_winners[i]
            ir_gt  = frame["gt"]
            rgb_gt = frame.get("rgb_gt")
            gt = rgb_gt if winner == "rgb" else ir_gt
        else:
            gt = frame["gt"]

        pred_empty = (pred is None or pred[2] <= 0 or pred[3] <= 0)
        p_t        = 1.0 if pred_empty else 0.0

        if gt is not None:
            iou_t = iou_xywh(pred, gt) if not pred_empty else 0.0
            reward_sum += iou_t
            if iou_t < failure_iou_thresh:
                failure_sum += 1.0
        else:
            reward_sum += p_t

    first_term = reward_sum / T
    penalty    = alpha * ((failure_sum / T) ** beta) if T > 0 else 0.0
    return first_term - penalty


# ---------------------------------------------------------------------------
# Core: replay one sequence with given thresholds
# ---------------------------------------------------------------------------

_EMPTY_BBOX = [0.0, 0.0, 0.0, 0.0]


def replay_sequence(log_rows, gt_frames,
                    theta_cls_detect, theta_loc_detect,
                    theta_cls_track, theta_loc_track):
    """
    Replay the DETECT/TRACK state machine against stored log data.

    Returns:
        preds_ir      : list of [x,y,w,h] scored against IR GT
        preds_dms     : list of [x,y,w,h] scored against DMS-winner GT
        dms_winners   : list of "ir"/"rgb" (stored dms_winner from log)
        switches_det  : number of DETECT->TRACK transitions in this replay
        switches_trk  : number of TRACK->DETECT reversions in this replay
        n_track       : total frames spent in TRACK
        n_detect      : total frames spent in DETECT
        diverged      : frames where replayed state != original logged state
    """
    n = min(len(log_rows), len(gt_frames))
    state = "DETECT"
    preds_ir = []
    preds_dms = []
    dms_winners = []
    switches_det = 0
    switches_trk = 0
    n_track = 0
    n_detect = 0
    diverged = 0

    for i in range(n):
        row = log_rows[i]
        u      = row["u"]
        sigma2 = row["sigma2"]
        orig   = row["orig_state"]

        # Count divergence vs original run
        if not orig.startswith(state):
            diverged += 1

        if state == "DETECT":
            n_detect += 1
            if u is not None and sigma2 is not None:
                if u < theta_cls_detect and sigma2 < theta_loc_detect:
                    state = "TRACK"
                    switches_det += 1

            # Always use the stored bboxes — the tracker outputs the YOLO
            # detection box in DETECT mode even when NOT switching (tracker.py
            # line 642).  Only frames where YOLO found nothing have empty bboxes.
            pred_dms = _select_bbox(row)
            pred_ir  = _ir_bbox(row)

        if state == "TRACK":
            n_track += 1
            if u is not None and sigma2 is not None:
                if not (u < theta_cls_track and sigma2 < theta_loc_track):
                    state = "DETECT"
                    switches_trk += 1

            pred_dms = _select_bbox(row)
            pred_ir  = _ir_bbox(row)

        preds_ir.append(pred_ir)
        preds_dms.append(pred_dms)
        dms_winners.append(row["dms_winner"] or "ir")

    return (preds_ir, preds_dms, dms_winners,
            switches_det, switches_trk, n_track, n_detect, diverged)


def _select_bbox(row):
    """Return the DMS-selected bbox from a log row, falling back to the other modality."""
    winner = (row.get("dms_winner") or "ir").strip()
    if winner == "rgb":
        bbox = row.get("rgb_bbox") or row.get("ir_bbox")
    else:
        bbox = row.get("ir_bbox") or row.get("rgb_bbox")
    return bbox if bbox is not None else _EMPTY_BBOX


def _ir_bbox(row):
    """Return the IR-specific bbox from a log row (always IR, never falls back to RGB).

    Matches the runner's acc_ir computation which uses ir_bbox exclusively,
    scored against IR GT.  Using the DMS-selected bbox (which may be RGB) for
    acc_ir produces near-zero IoU when the winner is RGB because the two camera
    coordinate spaces differ.
    """
    bbox = row.get("ir_bbox")
    return bbox if bbox is not None else _EMPTY_BBOX


# ---------------------------------------------------------------------------
# Load all logs and GT for a results dir
# ---------------------------------------------------------------------------

def load_all(log_dir: Path, dataset_root: Path):
    """Return list of (seq_name, log_rows, gt_frames) for every log in log_dir."""
    sequences = []
    for log_path in sorted(log_dir.glob("*_log.csv")):
        seq_name = log_path.stem[:-4]   # strip "_log"
        seq_dir  = dataset_root / seq_name
        if not seq_dir.is_dir():
            print(f"  [warn] GT dir not found for {seq_name}, skipping.")
            continue
        try:
            gt_frames = load_gt(str(seq_dir))
        except (FileNotFoundError, KeyError) as e:
            print(f"  [warn] GT load failed for {seq_name}: {e}")
            continue
        log_rows = load_log(log_path)
        if not log_rows or not gt_frames:
            continue
        sequences.append((seq_name, log_rows, gt_frames))
    return sequences


# ---------------------------------------------------------------------------
# Evaluate one threshold 4-tuple across all sequences
# ---------------------------------------------------------------------------

def evaluate_combo(sequences, combo, alpha, beta, failure_iou_thresh):
    theta_cls_detect, theta_loc_detect, theta_cls_track, theta_loc_track = combo

    acc_ir_list  = []
    acc_dms_list = []
    total_switches_det = 0
    total_switches_trk = 0
    total_track  = 0
    total_detect = 0
    total_frames = 0
    total_diverged = 0

    for seq_name, log_rows, gt_frames in sequences:
        (preds_ir, preds_dms, dms_winners,
         sw_det, sw_trk, n_trk, n_det, n_div) = replay_sequence(
            log_rows, gt_frames,
            theta_cls_detect, theta_loc_detect,
            theta_cls_track,  theta_loc_track,
        )
        n = min(len(log_rows), len(gt_frames))
        acc_ir  = compute_edtc_acc(preds_ir,  gt_frames[:n], alpha=alpha,
                                   beta=beta, failure_iou_thresh=failure_iou_thresh)
        acc_dms = compute_edtc_acc(preds_dms, gt_frames[:n], dms_winners=dms_winners,
                                   alpha=alpha, beta=beta,
                                   failure_iou_thresh=failure_iou_thresh)
        if math.isfinite(acc_ir):
            acc_ir_list.append(acc_ir)
        if math.isfinite(acc_dms):
            acc_dms_list.append(acc_dms)
        total_switches_det += sw_det
        total_switches_trk += sw_trk
        total_track        += n_trk
        total_detect       += n_det
        total_frames       += n
        total_diverged     += n_div

    mean_acc_ir  = float(sum(acc_ir_list)  / len(acc_ir_list))  if acc_ir_list  else float("nan")
    mean_acc_dms = float(sum(acc_dms_list) / len(acc_dms_list)) if acc_dms_list else float("nan")
    divergence_rate = total_diverged / total_frames if total_frames > 0 else 0.0
    thrash_rate  = total_switches_det / total_frames    if total_frames > 0 else 0.0
    revert_rate  = total_switches_trk / total_track     if total_track  > 0 else 0.0

    return {
        "theta_cls_detect": theta_cls_detect,
        "theta_loc_detect": theta_loc_detect,
        "theta_cls_track":  theta_cls_track,
        "theta_loc_track":  theta_loc_track,
        "acc_ir":           mean_acc_ir,
        "acc_dms":          mean_acc_dms,
        "switches_to_track":   total_switches_det,
        "switches_to_detect":  total_switches_trk,
        "frames_in_track":     total_track,
        "frames_in_detect":    total_detect,
        "thrash_rate":         thrash_rate,
        "revert_rate":         revert_rate,
        "divergence_rate":     divergence_rate,
        "n_sequences":         len(acc_ir_list),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Offline threshold calibration for EDTC (no GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log_dir",      required=True,
                   help="Directory containing *_log.csv files from a previous run.")
    p.add_argument("--dataset_root", required=True,
                   help="Anti-UAV-RGBT val split root (for GT loading).")
    p.add_argument("--out_dir",      default=None,
                   help="Where to write calibration_results.json / top10_candidates.json."
                        " Defaults to --log_dir.")

    p.add_argument("--theta_cls_detect", type=float, nargs="+", default=[0.80],
                   help="One or more values to sweep for theta_cls_detect.")
    p.add_argument("--theta_loc_detect", type=float, nargs="+", default=[0.10],
                   help="One or more values to sweep for theta_loc_detect.")
    p.add_argument("--theta_cls_track",  type=float, nargs="+", default=[0.90],
                   help="One or more values to sweep for theta_cls_track.")
    p.add_argument("--theta_loc_track",  type=float, nargs="+", default=[0.70],
                   help="One or more values to sweep for theta_loc_track.")

    p.add_argument("--alpha",              type=float, default=0.2)
    p.add_argument("--beta",               type=float, default=0.3)
    p.add_argument("--failure_iou_thresh", type=float, default=1e-5)

    p.add_argument("--thrash_thresh", type=float, default=0.30,
                   help="Reject combos where switches_to_track/total_frames exceeds this.")
    p.add_argument("--revert_thresh", type=float, default=0.20,
                   help="Reject combos where switches_to_detect/frames_in_track exceeds this.")
    p.add_argument("--rank_by", choices=["acc_ir", "acc_dms"], default="acc_ir",
                   help="Metric used to rank and select top candidates.")
    p.add_argument("--top_k", type=int, default=10,
                   help="How many top candidates to write to top<k>_candidates.json.")

    return p.parse_args()


def main():
    args = parse_args()
    log_dir      = Path(args.log_dir)
    dataset_root = Path(args.dataset_root)
    out_dir      = Path(args.out_dir) if args.out_dir else log_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading logs from   : {log_dir}")
    print(f"Loading GT from     : {dataset_root}")
    sequences = load_all(log_dir, dataset_root)
    if not sequences:
        raise SystemExit("No sequences loaded — check --log_dir and --dataset_root.")
    print(f"Loaded {len(sequences)} sequences.")

    combos = list(itertools.product(
        args.theta_cls_detect,
        args.theta_loc_detect,
        args.theta_cls_track,
        args.theta_loc_track,
    ))
    print(f"\nSweeping {len(combos)} threshold combinations...\n")

    results = []
    for idx, combo in enumerate(combos, 1):
        r = evaluate_combo(sequences, combo,
                           args.alpha, args.beta, args.failure_iou_thresh)
        # Switching frequency flags
        r["thrash_flag"] = r["thrash_rate"] > args.thrash_thresh
        r["revert_flag"] = r["revert_rate"] > args.revert_thresh
        results.append(r)

        tag = ""
        if r["thrash_flag"]: tag += " [THRASH]"
        if r["revert_flag"]: tag += " [REVERT]"
        if r["divergence_rate"] > 0.25: tag += " [HIGH-DIV]"
        print(
            f"[{idx:4d}/{len(combos)}]"
            f"  cls_d={combo[0]:.2f}  loc_d={combo[1]:.3f}"
            f"  cls_t={combo[2]:.2f}  loc_t={combo[3]:.2f}"
            f"  acc_ir={r['acc_ir']:.4f}  acc_dms={r['acc_dms']:.4f}"
            f"  div={r['divergence_rate']:.2f}"
            f"{tag}"
        )

    # Rank by chosen metric, passing entries first, then flagged ones
    def sort_key(r):
        flagged = int(r["thrash_flag"]) + int(r["revert_flag"])
        score   = r[args.rank_by] if math.isfinite(r[args.rank_by]) else -999.0
        return (flagged, -score)

    results.sort(key=sort_key)

    # Save full results
    results_path = out_dir / "calibration_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results -> {results_path}")

    # Save top-K
    top_k = [r for r in results if not r["thrash_flag"] and not r["revert_flag"]][:args.top_k]
    if len(top_k) < args.top_k:
        top_k += [r for r in results if r not in top_k][: args.top_k - len(top_k)]

    top_path = out_dir / f"top{args.top_k}_candidates.json"
    with open(top_path, "w") as f:
        json.dump(top_k, f, indent=2)
    print(f"Top-{args.top_k} candidates -> {top_path}")

    print(f"\n{'='*70}")
    print(f"TOP-5 (ranked by {args.rank_by}):")
    for i, r in enumerate(top_k[:5], 1):
        flags = ""
        if r["thrash_flag"]: flags += " THRASH"
        if r["revert_flag"]: flags += " REVERT"
        if r["divergence_rate"] > 0.25: flags += " HIGH-DIV"
        print(
            f"  {i}. cls_d={r['theta_cls_detect']:.2f}  loc_d={r['theta_loc_detect']:.3f}"
            f"  cls_t={r['theta_cls_track']:.2f}  loc_t={r['theta_loc_track']:.2f}"
            f"  acc_ir={r['acc_ir']:.4f}  acc_dms={r['acc_dms']:.4f}"
            f"  div={r['divergence_rate']:.2f}{flags}"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
