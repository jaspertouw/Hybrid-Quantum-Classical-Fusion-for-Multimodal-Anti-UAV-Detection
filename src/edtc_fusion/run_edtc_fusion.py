"""
run_edtc_fusion.py — Phase 6.1 sanity-check runner
==================================================
Run the EDTCFusionTracker on the Anti-UAV RGB-T validation split with fixed
switching thresholds, and report EDTC Acc per the standard formula:

    Acc = [ Σ_t  IoU_t·δ(v_t>0) + p_t·(1−δ(v_t>0)) ] / T
          − α · ( Σ_{t: v_t>0} q_t / T* )^β

Per-sequence outputs (under --results_dir):
    <seq>.txt           : predicted [x, y, w, h] per frame (tab-separated)
    <seq>_log.csv       : per-frame state, u_fused, sigma2_best, switched flag
    summary.json        : aggregate Acc, mean switching counts, threshold values

Usage (Snellius)::

    python -m src.edtc_fusion.run_edtc_fusion \\
        --dataset_root /home/ssuttorp/Anti-UAV-RGBT/val \\
        --yolo_ir   /home/ssuttorp/pretrained_models/evidential/evidential_ir_stage2_best.pt  \\
        --yolo_rgb  /home/ssuttorp/pretrained_models/evidential/evidential_rgb_stage2_best.pt \\
        --yolo_ir_base  /home/ssuttorp/pretrained_models/best_clean.pt     \\
        --yolo_rgb_base /home/ssuttorp/pretrained_models/best_clean_rgb.pt \\
        --siam_ir_stage2  /home/ssuttorp/pretrained_models/siamcar_gaussian/siamcar_ir_stage2_best.pth  \\
        --siam_rgb_stage2 /home/ssuttorp/pretrained_models/siamcar_gaussian/siamcar_rgb_stage2_best.pth \\
        --siam_ir_stage1  /home/ssuttorp/pretrained_models/siamcar_gaussian/siamcar_ir_stage1g_best.pth \\
        --siam_rgb_stage1 /home/ssuttorp/pretrained_models/siamcar_gaussian/siamcar_rgb_stage1g_best.pth \\
        --siam_ir_pretrained  /home/ssuttorp/pretrained_models/siamcar_thermal_final.pth \\
        --siam_rgb_pretrained /home/ssuttorp/pretrained_models/siamcar_rgb_final.pth     \\
        --siam_ir_cfg  resources/SiamCAR/experiments/siamcar_r50/config_antiuav_thermal.yaml \\
        --siam_rgb_cfg resources/SiamCAR/experiments/siamcar_r50/config_antiuav_rgb.yaml     \\
        --results_dir /home/ssuttorp/pretrained_models/edtc_fusion/val_results              \\
        --fusion dbf
"""

from __future__ import absolute_import, annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.edtc_fusion.tracker import EDTCFusionTracker, FusionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edtc_fusion")


# ===========================================================================
# Sequence loader — pairs IR and visible frames using the two label JSONs
# ===========================================================================

def load_paired_sequence(seq_dir: str):
    """Return list of dicts with paired frame paths and IR ground-truth.

    The Anti-UAV-RGBT dataset stores per-modality labels (``IR_label.json`` and
    ``visible_label.json``) and per-modality frames (``ir_frames/`` and
    ``rgb_frames/``).  Sequences are not always equal length — we use the IR
    track as the canonical timeline and skip any frame missing from RGB.
    """
    ir_label_path  = os.path.join(seq_dir, "IR_label.json")
    rgb_label_path = os.path.join(seq_dir, "visible_label.json")

    with open(ir_label_path)  as f: ir_ann  = json.load(f)
    with open(rgb_label_path) as f: rgb_ann = json.load(f)

    n = min(len(ir_ann["exist"]), len(rgb_ann["exist"]))
    ir_dir  = os.path.join(seq_dir, "ir_frames")
    rgb_dir = os.path.join(seq_dir, "rgb_frames")

    frames = []
    for i in range(n):
        fname = "{:06d}.jpg".format(i + 1)
        ir_path  = os.path.join(ir_dir,  fname)
        rgb_path = os.path.join(rgb_dir, fname)
        if not (os.path.exists(ir_path) and os.path.exists(rgb_path)):
            continue
        gt      = ir_ann["gt_rect"][i]  if ir_ann["exist"][i]  else None
        rgb_gt  = rgb_ann["gt_rect"][i] if rgb_ann["exist"][i] else None
        frames.append({"ir": ir_path, "rgb": rgb_path, "gt": gt, "rgb_gt": rgb_gt})
    return frames


# ===========================================================================
# Metrics — EDTC Acc (matches eval_antiuav_thermal.py exactly)
# ===========================================================================

def iou_xywh(pred, gt):
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px + pw, gx + gw), min(py + ph, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return inter / (union + 1e-8)


def compute_edtc_acc(preds, frames, alpha=0.2, beta=0.3, failure_iou_thresh=1e-5):
    T = len(frames)
    if T == 0:
        return float("nan")

    reward_sum, failure_sum = 0.0, 0.0
    for pred, frame in zip(preds, frames):
        gt = frame["gt"]
        pred_empty = (pred[2] <= 0 or pred[3] <= 0)
        p_t = 1.0 if pred_empty else 0.0

        if gt is not None:
            iou_t = iou_xywh(pred, gt) if not pred_empty else 0.0
            reward_sum += iou_t
            if iou_t < failure_iou_thresh:
                failure_sum += 1.0
        else:
            reward_sum += p_t

    first_term = reward_sum / T
    penalty = alpha * ((failure_sum / T) ** beta) if T > 0 else 0.0
    return first_term - penalty


def compute_edtc_acc_dms(preds, frames, dms_winners,
                         alpha=0.2, beta=0.3, failure_iou_thresh=1e-5):
    """Option-B DMS-aware Acc: per-frame GT is selected to match the winning
    modality, so the prediction is always measured in its own coordinate space.

    Existence (T*) is anchored to IR GT — thermal IR is the primary detection
    modality and IR_label.json is the authoritative presence/absence signal.
    When the UAV is present by IR GT, IoU is computed against IR GT (winner='ir')
    or RGB GT (winner='rgb'); if the winner is RGB but rgb_gt is unavailable for
    that frame, IoU falls back to 0.
    """
    T = len(frames)
    if T == 0:
        return float("nan")

    reward_sum, failure_sum = 0.0, 0.0
    for pred, frame, winner in zip(preds, frames, dms_winners):
        ir_gt  = frame["gt"]              # None ↔ UAV absent (IR primary)
        rgb_gt = frame.get("rgb_gt")      # None ↔ UAV absent in RGB GT

        pred_empty = (pred[2] <= 0 or pred[3] <= 0)
        p_t        = 1.0 if pred_empty else 0.0

        if ir_gt is not None:             # UAV present — compute IoU
            gt_for_iou = rgb_gt if winner == "rgb" else ir_gt
            if gt_for_iou is not None and not pred_empty:
                iou_t = iou_xywh(pred, gt_for_iou)
            else:
                iou_t = 0.0
            reward_sum += iou_t
            if iou_t < failure_iou_thresh:
                failure_sum += 1.0
        else:                             # UAV absent — score absence prediction
            reward_sum += p_t

    first_term = reward_sum / T
    penalty    = alpha * ((failure_sum / T) ** beta) if T > 0 else 0.0
    return first_term - penalty


# ===========================================================================
# Per-sequence runner
# ===========================================================================

def run_sequence(tracker: EDTCFusionTracker, frames):
    """Run tracker over one paired sequence.  Returns per-frame predictions."""
    tracker.reset()
    preds = []
    for fr in frames:
        ir_img  = cv2.imread(fr["ir"])    # BGR
        rgb_img = cv2.imread(fr["rgb"])   # BGR
        if ir_img is None or rgb_img is None:
            preds.append([0.0, 0.0, 0.0, 0.0])
            continue
        out = tracker.step(ir_img, rgb_img)
        preds.append(out["bbox"])
    return preds


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Anti-UAV — EDTC fusion (Phase 6.1) sanity check")

    # Model paths
    p.add_argument("--yolo_ir",   required=True)
    p.add_argument("--yolo_rgb",  required=True)
    p.add_argument("--yolo_ir_base",  required=True, help="original YOLOv5 .pt for IR (architecture)")
    p.add_argument("--yolo_rgb_base", required=True, help="original YOLOv5 .pt for RGB (architecture)")

    p.add_argument("--siam_ir_stage2",  required=True)
    p.add_argument("--siam_rgb_stage2", required=True)
    p.add_argument("--siam_ir_stage1",  required=True)
    p.add_argument("--siam_rgb_stage1", required=True)
    p.add_argument("--siam_ir_pretrained",  required=True)
    p.add_argument("--siam_rgb_pretrained", required=True)
    p.add_argument("--siam_ir_cfg",  required=True)
    p.add_argument("--siam_rgb_cfg", required=True)

    # Dataset + output
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--results_dir",  required=True)
    p.add_argument("--sequence", default=None,
                   help="Run a single sequence by name instead of the whole split.")

    # Fusion + thresholds (sanity-check defaults)
    p.add_argument("--fusion", choices=["dbf", "abf"], default="dbf")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--theta_cls_detect", type=float, default=0.80,
                   help="u_fused upper bound for DETECT->TRACK (empirical: floor ~0.65)")
    p.add_argument("--theta_loc_detect", type=float, default=0.10,
                   help="sigma2_best upper bound for DETECT->TRACK (empirical: ~0.04)")
    p.add_argument("--theta_cls_track",  type=float, default=0.90,
                   help="u_fused upper bound for TRACK maintenance (SiamCAR peak u: 0.16–0.61)")
    p.add_argument("--theta_loc_track",  type=float, default=0.70,
                   help="sigma2_best upper bound for TRACK maintenance. "
                        "Empirically: good-tracking sigma2 ≈ 0.46–0.68; drift first "
                        "exceeds 0.72 (smoke seq frame 5). 0.70 catches the drift; "
                        "previous value 2.00 was inert — drifted tracker ghost-tracks "
                        "background at sigma2 ≈ 0.5–0.9.")

    # YOLO knobs
    p.add_argument("--yolo_imgsz",   type=int,   default=640)
    p.add_argument("--yolo_conf",    type=float, default=0.25)
    p.add_argument("--yolo_iou",     type=float, default=0.45)
    p.add_argument("--yolo_classes", type=int,   nargs="*", default=None,
                   help="NMS class filter (default: [0] = UAV only). "
                        "Pass nothing ('--yolo_classes') for no filtering.")

    # YOLO uncertainty aggregation (see FusionConfig.unc_agg). Affects the
    # DETECT phase only — SiamCAR tracking uncertainty is always peak-sampled.
    p.add_argument("--unc_agg", choices=["mean", "peak"], default="mean",
                   help="'peak' = belief/uncertainty of the selected detection only; "
                        "'mean' = average over all pre-NMS anchors (legacy default).")

    # Missing-detection encoding for a YOLO stream that returns nothing.
    p.add_argument("--miss_belief", choices=["dogmatic_bg", "vacuous"],
                   default="dogmatic_bg",
                   help="Belief assigned to a YOLO stream whose NMS returns "
                        "nothing. 'dogmatic_bg' = legacy _ZERO_UNC placeholder "
                        "b=(0,1), u=1; 'vacuous' = partition-consistent "
                        "zero-evidence opinion b=(0,0), u=1 (corrected main "
                        "results).")

    # EDTC Acc
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta",  type=float, default=0.3)
    p.add_argument("--failure_iou_thresh", type=float, default=1e-5)

    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def discover_sequences(root):
    list_file = os.path.join(root, "list.txt")
    if os.path.exists(list_file):
        with open(list_file) as f:
            return [s.strip() for s in f if s.strip()]
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def main():
    args = parse_args()

    cfg = FusionConfig(
        yolo_ir_weights=args.yolo_ir,
        yolo_rgb_weights=args.yolo_rgb,
        yolo_ir_base=args.yolo_ir_base,
        yolo_rgb_base=args.yolo_rgb_base,
        siam_ir_stage2=args.siam_ir_stage2,
        siam_rgb_stage2=args.siam_rgb_stage2,
        siam_ir_stage1=args.siam_ir_stage1,
        siam_rgb_stage1=args.siam_rgb_stage1,
        siam_ir_pretrained=args.siam_ir_pretrained,
        siam_rgb_pretrained=args.siam_rgb_pretrained,
        siam_ir_cfg=args.siam_ir_cfg,
        siam_rgb_cfg=args.siam_rgb_cfg,
        fusion=args.fusion,
        lam=args.lam,
        theta_cls_detect=args.theta_cls_detect,
        theta_loc_detect=args.theta_loc_detect,
        theta_cls_track=args.theta_cls_track,
        theta_loc_track=args.theta_loc_track,
        yolo_imgsz=(args.yolo_imgsz, args.yolo_imgsz),
        yolo_conf_thres=args.yolo_conf,
        yolo_iou_thres=args.yolo_iou,
        yolo_classes=args.yolo_classes,   # None → __post_init__ sets [0] (UAV only)
        unc_agg=args.unc_agg,
        miss_belief=args.miss_belief,
        device=args.device,
    )

    logger.info("Building EDTCFusionTracker...")
    tracker = EDTCFusionTracker(cfg)

    sequences = discover_sequences(args.dataset_root)
    if args.sequence is not None:
        sequences = [s for s in sequences if s == args.sequence]
        if not sequences:
            raise SystemExit(f'Sequence "{args.sequence}" not found.')

    os.makedirs(args.results_dir, exist_ok=True)

    ir_accs  = []   # Acc_IR  : IR tracker predictions vs IR GT  (Option A)
    dms_accs = []   # Acc_DMS : DMS-selected prediction vs winner GT (Option B)
    n_switch_to_track = 0
    n_switch_to_detect = 0
    total_track_frames = 0
    total_detect_frames = 0

    for v_idx, seq_name in enumerate(sequences):
        seq_dir = os.path.join(args.dataset_root, seq_name)
        frames  = load_paired_sequence(seq_dir)
        if not frames:
            logger.warning("No paired frames for %s — skipping.", seq_name)
            continue

        preds = run_sequence(tracker, frames)

        # Telemetry from tracker.frame_log (one entry per processed frame)
        seq_log      = tracker.frame_log
        # IR-only predictions: use ir_bbox from log (Option-A Acc_IR)
        ir_preds     = [r["ir_bbox"] if r.get("ir_bbox") else [0.0, 0.0, 0.0, 0.0]
                        for r in seq_log]
        # DMS winner per frame for Option-B routing (default 'ir' if missing)
        dms_winners  = [r.get("dms_winner") or "ir" for r in seq_log]

        acc_ir = compute_edtc_acc(ir_preds, frames,
                                  alpha=args.alpha, beta=args.beta,
                                  failure_iou_thresh=args.failure_iou_thresh)
        acc_dms = compute_edtc_acc_dms(preds, frames, dms_winners,
                                       alpha=args.alpha, beta=args.beta,
                                       failure_iou_thresh=args.failure_iou_thresh)
        if not math.isnan(acc_ir):
            ir_accs.append(acc_ir)
        if not math.isnan(acc_dms):
            dms_accs.append(acc_dms)

        seq_n_to_track  = sum(1 for r in seq_log if r["state"] == "DETECT" and r["switched"])
        seq_n_to_detect = sum(1 for r in seq_log if r["state"] == "TRACK"  and r["switched"])
        seq_n_track_fr  = sum(1 for r in seq_log if r["state"] == "TRACK")
        seq_n_detect_fr = sum(1 for r in seq_log if r["state"] == "DETECT")
        n_switch_to_track  += seq_n_to_track
        n_switch_to_detect += seq_n_to_detect
        total_track_frames += seq_n_track_fr
        total_detect_frames += seq_n_detect_fr

        logger.info(
            "[%3d/%3d] %-30s  Acc_IR=%.4f  Acc_DMS=%.4f  "
            "TRACK_fr=%d  DETECT_fr=%d  to_track=%d  to_detect=%d",
            v_idx + 1, len(sequences), seq_name, acc_ir, acc_dms,
            seq_n_track_fr, seq_n_detect_fr, seq_n_to_track, seq_n_to_detect,
        )

        # Save DMS-selected bbox predictions (winner's coordinate space per frame)
        bbox_path = os.path.join(args.results_dir, seq_name + ".txt")
        with open(bbox_path, "w") as f:
            for bbox in preds:
                f.write(",".join("{:.4f}".format(v) for v in bbox) + "\n")

        # Save IR-only predictions (always in IR pixel space)
        ir_bbox_path = os.path.join(args.results_dir, seq_name + "_ir.txt")
        with open(ir_bbox_path, "w") as f:
            for bbox in ir_preds:
                f.write(",".join("{:.4f}".format(v) for v in bbox) + "\n")

        # Save per-frame log (includes ir_bbox, rgb_bbox, dms_winner for audit)
        log_path = os.path.join(args.results_dir, seq_name + "_log.csv")
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "state", "u_fused", "b_uav", "sigma2_best", "switched",
                        "ir_bbox", "rgb_bbox", "dms_winner"])
            for i, r in enumerate(seq_log):
                ir_b  = json.dumps(r["ir_bbox"])  if r.get("ir_bbox")  else ""
                rgb_b = json.dumps(r["rgb_bbox"]) if r.get("rgb_bbox") else ""
                w.writerow([
                    i, r["state"],
                    r["u_fused"]     if r["u_fused"]     is not None else "",
                    r["b_uav"]       if r.get("b_uav")   is not None else "",
                    r["sigma2_best"] if r["sigma2_best"] is not None else "",
                    int(r["switched"]) if r["switched"] is not None else "",
                    ir_b, rgb_b, r.get("dms_winner", ""),
                ])

    mean_acc_ir  = float(np.mean(ir_accs))  if ir_accs  else float("nan")
    mean_acc_dms = float(np.mean(dms_accs)) if dms_accs else float("nan")
    logger.info("=" * 70)
    logger.info("Acc_IR  (IR preds vs IR GT)           = %.4f  [%d sequences]",
                mean_acc_ir,  len(ir_accs))
    logger.info("Acc_DMS (DMS-selected pred vs winner GT) = %.4f  [%d sequences]",
                mean_acc_dms, len(dms_accs))
    logger.info("Switches DETECT->TRACK : %d", n_switch_to_track)
    logger.info("Switches TRACK->DETECT : %d", n_switch_to_detect)
    logger.info("Frames in TRACK  : %d", total_track_frames)
    logger.info("Frames in DETECT : %d", total_detect_frames)
    logger.info(
        "thresholds: cls_det=%.3f loc_det=%.3f cls_trk=%.3f loc_trk=%.3f  fusion=%s",
        cfg.theta_cls_detect, cfg.theta_loc_detect,
        cfg.theta_cls_track,  cfg.theta_loc_track, cfg.fusion,
    )
    logger.info("=" * 70)

    summary = {
        "acc_ir":              mean_acc_ir,
        "acc_dms":             mean_acc_dms,
        "n_sequences":         len(ir_accs),
        "switches_to_track":   n_switch_to_track,
        "switches_to_detect":  n_switch_to_detect,
        "frames_in_track":     total_track_frames,
        "frames_in_detect":    total_detect_frames,
        "thresholds": {
            "theta_cls_detect": cfg.theta_cls_detect,
            "theta_loc_detect": cfg.theta_loc_detect,
            "theta_cls_track":  cfg.theta_cls_track,
            "theta_loc_track":  cfg.theta_loc_track,
        },
        "fusion": cfg.fusion,
        "lam":    cfg.lam,
        "unc_agg":     cfg.unc_agg,
        "miss_belief": cfg.miss_belief,
        "alpha":  args.alpha,
        "beta":   args.beta,
        "failure_iou_thresh": args.failure_iou_thresh,
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary -> %s", os.path.join(args.results_dir, "summary.json"))


if __name__ == "__main__":
    main()
