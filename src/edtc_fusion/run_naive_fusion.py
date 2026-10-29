"""
run_naive_fusion.py — Naive Late Fusion evaluation runner
=========================================================
Evaluates EDTCNaiveFusionTracker on the Anti-UAV RGB-T validation split.
Identical to run_edtc_fusion.py except:
  - uses EDTCNaiveFusionTracker / NaiveFusionConfig
  - no --fusion / --lam arguments
  - summary.json records method="naive_late_fusion"

Usage (Snellius)::

    python -m src.edtc_fusion.run_naive_fusion \\
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
        --results_dir /home/ssuttorp/pretrained_models/naive_fusion/val_results
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

from src.edtc_fusion.tracker_naive_fusion import EDTCNaiveFusionTracker, NaiveFusionConfig
from src.edtc_fusion.run_edtc_fusion import (
    load_paired_sequence,
    compute_edtc_acc,
    compute_edtc_acc_dms,
    discover_sequences,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("naive_fusion")


def run_sequence(tracker: EDTCNaiveFusionTracker, frames):
    tracker.reset()
    preds = []
    for fr in frames:
        ir_img  = cv2.imread(fr["ir"])
        rgb_img = cv2.imread(fr["rgb"])
        if ir_img is None or rgb_img is None:
            preds.append([0.0, 0.0, 0.0, 0.0])
            continue
        out = tracker.step(ir_img, rgb_img)
        preds.append(out["bbox"])
    return preds


def parse_args():
    p = argparse.ArgumentParser(description="Anti-UAV — Naive Late Fusion evaluation")

    p.add_argument("--yolo_ir",   required=True)
    p.add_argument("--yolo_rgb",  required=True)
    p.add_argument("--yolo_ir_base",  required=True)
    p.add_argument("--yolo_rgb_base", required=True)

    p.add_argument("--siam_ir_stage2",  required=True)
    p.add_argument("--siam_rgb_stage2", required=True)
    p.add_argument("--siam_ir_stage1",  required=True)
    p.add_argument("--siam_rgb_stage1", required=True)
    p.add_argument("--siam_ir_pretrained",  required=True)
    p.add_argument("--siam_rgb_pretrained", required=True)
    p.add_argument("--siam_ir_cfg",  required=True)
    p.add_argument("--siam_rgb_cfg", required=True)

    p.add_argument("--dataset_root", required=True)
    p.add_argument("--results_dir",  required=True)
    p.add_argument("--sequence", default=None)

    p.add_argument("--theta_cls_detect", type=float, default=0.80)
    p.add_argument("--theta_loc_detect", type=float, default=0.10)
    p.add_argument("--theta_cls_track",  type=float, default=0.90)
    p.add_argument("--theta_loc_track",  type=float, default=0.70)

    p.add_argument("--yolo_imgsz",   type=int,   default=640)
    p.add_argument("--yolo_conf",    type=float, default=0.25)
    p.add_argument("--yolo_iou",     type=float, default=0.45)
    p.add_argument("--yolo_classes", type=int,   nargs="*", default=None)

    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta",  type=float, default=0.3)
    p.add_argument("--failure_iou_thresh", type=float, default=1e-5)

    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = NaiveFusionConfig(
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
        theta_cls_detect=args.theta_cls_detect,
        theta_loc_detect=args.theta_loc_detect,
        theta_cls_track=args.theta_cls_track,
        theta_loc_track=args.theta_loc_track,
        yolo_imgsz=(args.yolo_imgsz, args.yolo_imgsz),
        yolo_conf_thres=args.yolo_conf,
        yolo_iou_thres=args.yolo_iou,
        yolo_classes=args.yolo_classes,
        device=args.device,
    )

    logger.info("Building EDTCNaiveFusionTracker...")
    tracker = EDTCNaiveFusionTracker(cfg)

    sequences = discover_sequences(args.dataset_root)
    if args.sequence is not None:
        sequences = [s for s in sequences if s == args.sequence]
        if not sequences:
            raise SystemExit(f'Sequence "{args.sequence}" not found.')

    os.makedirs(args.results_dir, exist_ok=True)

    ir_accs  = []
    dms_accs = []
    n_switch_to_track  = 0
    n_switch_to_detect = 0
    total_track_frames  = 0
    total_detect_frames = 0

    for v_idx, seq_name in enumerate(sequences):
        seq_dir = os.path.join(args.dataset_root, seq_name)
        frames  = load_paired_sequence(seq_dir)
        if not frames:
            logger.warning("No paired frames for %s — skipping.", seq_name)
            continue

        preds = run_sequence(tracker, frames)

        seq_log     = tracker.frame_log
        ir_preds    = [r["ir_bbox"] if r.get("ir_bbox") else [0.0, 0.0, 0.0, 0.0]
                       for r in seq_log]
        dms_winners = [r.get("dms_winner") or "ir" for r in seq_log]

        acc_ir  = compute_edtc_acc(ir_preds, frames,
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
        total_track_frames  += seq_n_track_fr
        total_detect_frames += seq_n_detect_fr

        logger.info(
            "[%3d/%3d] %-30s  Acc_IR=%.4f  Acc_DMS=%.4f  "
            "TRACK_fr=%d  DETECT_fr=%d  to_track=%d  to_detect=%d",
            v_idx + 1, len(sequences), seq_name, acc_ir, acc_dms,
            seq_n_track_fr, seq_n_detect_fr, seq_n_to_track, seq_n_to_detect,
        )

        bbox_path = os.path.join(args.results_dir, seq_name + ".txt")
        with open(bbox_path, "w") as f:
            for bbox in preds:
                f.write(",".join("{:.4f}".format(v) for v in bbox) + "\n")

        ir_bbox_path = os.path.join(args.results_dir, seq_name + "_ir.txt")
        with open(ir_bbox_path, "w") as f:
            for bbox in ir_preds:
                f.write(",".join("{:.4f}".format(v) for v in bbox) + "\n")

        log_path = os.path.join(args.results_dir, seq_name + "_log.csv")
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "state", "u_fused", "sigma2_best", "switched",
                        "ir_bbox", "rgb_bbox", "dms_winner"])
            for i, r in enumerate(seq_log):
                ir_b  = json.dumps(r["ir_bbox"])  if r.get("ir_bbox")  else ""
                rgb_b = json.dumps(r["rgb_bbox"]) if r.get("rgb_bbox") else ""
                w.writerow([
                    i, r["state"],
                    r["u_fused"]     if r["u_fused"]     is not None else "",
                    r["sigma2_best"] if r["sigma2_best"] is not None else "",
                    int(r["switched"]) if r["switched"] is not None else "",
                    ir_b, rgb_b, r.get("dms_winner", ""),
                ])

    mean_acc_ir  = float(np.mean(ir_accs))  if ir_accs  else float("nan")
    mean_acc_dms = float(np.mean(dms_accs)) if dms_accs else float("nan")
    logger.info("=" * 70)
    logger.info("Acc_IR  (IR preds vs IR GT)              = %.4f  [%d sequences]",
                mean_acc_ir,  len(ir_accs))
    logger.info("Acc_DMS (DMS-selected pred vs winner GT) = %.4f  [%d sequences]",
                mean_acc_dms, len(dms_accs))
    logger.info("Switches DETECT->TRACK : %d", n_switch_to_track)
    logger.info("Switches TRACK->DETECT : %d", n_switch_to_detect)
    logger.info("Frames in TRACK  : %d", total_track_frames)
    logger.info("Frames in DETECT : %d", total_detect_frames)
    logger.info(
        "thresholds: cls_det=%.3f loc_det=%.3f cls_trk=%.3f loc_trk=%.3f  method=naive_late_fusion",
        cfg.theta_cls_detect, cfg.theta_loc_detect,
        cfg.theta_cls_track,  cfg.theta_loc_track,
    )
    logger.info("=" * 70)

    summary = {
        "method":              "naive_late_fusion",
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
        "alpha":  args.alpha,
        "beta":   args.beta,
        "failure_iou_thresh": args.failure_iou_thresh,
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary -> %s", os.path.join(args.results_dir, "summary.json"))


if __name__ == "__main__":
    main()
