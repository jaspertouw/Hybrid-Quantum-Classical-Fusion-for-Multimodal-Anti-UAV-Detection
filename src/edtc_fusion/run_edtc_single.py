"""
run_edtc_single.py — Single-modality EDTC baseline runner
=========================================================
Runs EDTCSingleTracker on the Anti-UAV RGB-T validation split for ONE
modality (thermal-only or RGB-only) and reports EDTC Acc.  This is the
single-modality baseline against which the DBF and ABF fusion runs are
compared.

Per-sequence outputs (under --results_dir):
    <seq>.txt       : predicted [x, y, w, h] per frame
    <seq>_log.csv   : per-frame state, u, sigma2, switched flag
    summary.json    : aggregate Acc, switching counts, thresholds

Usage:
    python -m src.edtc_fusion.run_edtc_single \
        --modality        ir  \
        --dataset_root    /path/to/Anti-UAV-RGBT/val      \
        --yolo            .../evidential_ir_stage2_best.pt \
        --yolo_base       .../best_clean.pt                \
        --siam_stage2     .../siamcar_ir_stage2_best.pth   \
        --siam_stage1     .../siamcar_ir_stage1g_best.pth  \
        --siam_pretrained .../siamcar_thermal_final.pth    \
        --siam_cfg        resources/SiamCAR/experiments/siamcar_r50/config_antiuav_thermal.yaml \
        --results_dir     /path/to/out
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

from src.edtc_fusion.tracker_single import EDTCSingleTracker, SingleConfig
from src.edtc_fusion.run_edtc_fusion import (
    compute_edtc_acc,
    load_paired_sequence,
    discover_sequences,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edtc_single")


def run_sequence(tracker: EDTCSingleTracker, frames, modality: str):
    """Run tracker over one paired sequence using only the chosen modality's frames."""
    tracker.reset()
    key = "ir" if modality == "ir" else "rgb"
    preds = []
    for fr in frames:
        img = cv2.imread(fr[key])
        if img is None:
            preds.append([0.0, 0.0, 0.0, 0.0])
            continue
        out = tracker.step(img)
        preds.append(out["bbox"])
    return preds


def parse_args():
    p = argparse.ArgumentParser(description="Anti-UAV — single-modality EDTC baseline")

    p.add_argument("--modality", choices=["ir", "rgb"], required=True)

    p.add_argument("--yolo",            required=True)
    p.add_argument("--yolo_base",       required=True,
                   help="original YOLOv5 .pt used to reconstruct architecture")
    p.add_argument("--siam_stage2",     required=True)
    p.add_argument("--siam_stage1",     required=True)
    p.add_argument("--siam_pretrained", required=True)
    p.add_argument("--siam_cfg",        required=True)

    p.add_argument("--dataset_root", required=True)
    p.add_argument("--results_dir",  required=True)
    p.add_argument("--sequence", default=None,
                   help="Run a single sequence by name instead of the whole split.")

    # Switching thresholds (defaults match fusion config)
    p.add_argument("--theta_cls_detect", type=float, default=0.80)
    p.add_argument("--theta_loc_detect", type=float, default=0.10)
    p.add_argument("--theta_cls_track",  type=float, default=0.90)
    p.add_argument("--theta_loc_track",  type=float, default=0.70)

    # YOLO knobs
    p.add_argument("--yolo_imgsz",   type=int,   default=640)
    p.add_argument("--yolo_conf",    type=float, default=0.25)
    p.add_argument("--yolo_iou",     type=float, default=0.45)
    p.add_argument("--yolo_classes", type=int,   nargs="*", default=None)

    # YOLO uncertainty aggregation (see FusionConfig.unc_agg). No miss_belief
    # here — single-stream has no fusion, so a YOLO miss just yields an empty box.
    p.add_argument("--unc_agg", choices=["mean", "peak"], default="mean",
                   help="'peak' = uncertainty of the selected detection only; "
                        "'mean' = average over all pre-NMS anchors (legacy default).")

    # EDTC Acc
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta",  type=float, default=0.3)
    p.add_argument("--failure_iou_thresh", type=float, default=1e-5)

    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = SingleConfig(
        modality=args.modality,
        yolo_weights=args.yolo,
        yolo_base=args.yolo_base,
        siam_stage2=args.siam_stage2,
        siam_stage1=args.siam_stage1,
        siam_pretrained=args.siam_pretrained,
        siam_cfg=args.siam_cfg,
        theta_cls_detect=args.theta_cls_detect,
        theta_loc_detect=args.theta_loc_detect,
        theta_cls_track=args.theta_cls_track,
        theta_loc_track=args.theta_loc_track,
        yolo_imgsz=(args.yolo_imgsz, args.yolo_imgsz),
        yolo_conf_thres=args.yolo_conf,
        yolo_iou_thres=args.yolo_iou,
        yolo_classes=args.yolo_classes,
        unc_agg=args.unc_agg,
        device=args.device,
    )

    logger.info("Building EDTCSingleTracker (modality=%s)...", args.modality)
    tracker = EDTCSingleTracker(cfg)

    sequences = discover_sequences(args.dataset_root)
    if args.sequence is not None:
        sequences = [s for s in sequences if s == args.sequence]
        if not sequences:
            raise SystemExit(f'Sequence "{args.sequence}" not found.')

    os.makedirs(args.results_dir, exist_ok=True)

    # For single-modality, evaluate against the matching modality's GT.
    # load_paired_sequence keeps both labels; pick the right one.
    gt_key = "gt" if args.modality == "ir" else "rgb_gt"

    accs = []
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

        preds = run_sequence(tracker, frames, args.modality)

        # Build a per-modality view of frames for Acc (gt swap when modality=rgb)
        eval_frames = [{"gt": fr[gt_key]} for fr in frames]
        acc = compute_edtc_acc(preds, eval_frames,
                               alpha=args.alpha, beta=args.beta,
                               failure_iou_thresh=args.failure_iou_thresh)
        if not math.isnan(acc):
            accs.append(acc)

        seq_log = tracker.frame_log
        seq_n_to_track  = sum(1 for r in seq_log if r["state"] == "DETECT" and r["switched"])
        seq_n_to_detect = sum(1 for r in seq_log if r["state"] == "TRACK"  and r["switched"])
        seq_n_track_fr  = sum(1 for r in seq_log if r["state"] == "TRACK")
        seq_n_detect_fr = sum(1 for r in seq_log if r["state"].startswith("DETECT"))
        n_switch_to_track  += seq_n_to_track
        n_switch_to_detect += seq_n_to_detect
        total_track_frames += seq_n_track_fr
        total_detect_frames += seq_n_detect_fr

        logger.info(
            "[%3d/%3d] %-30s  Acc=%.4f  TRACK_fr=%d  DETECT_fr=%d  "
            "to_track=%d  to_detect=%d",
            v_idx + 1, len(sequences), seq_name, acc,
            seq_n_track_fr, seq_n_detect_fr, seq_n_to_track, seq_n_to_detect,
        )

        # Predictions
        bbox_path = os.path.join(args.results_dir, seq_name + ".txt")
        with open(bbox_path, "w") as f:
            for bbox in preds:
                f.write(",".join("{:.4f}".format(v) for v in bbox) + "\n")

        # Per-frame log
        log_path = os.path.join(args.results_dir, seq_name + "_log.csv")
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "state", "u", "b_uav", "sigma2", "switched", "bbox"])
            for i, r in enumerate(seq_log):
                w.writerow([
                    i, r["state"],
                    r["u"]      if r["u"]      is not None else "",
                    r["b_uav"]  if r.get("b_uav") is not None else "",
                    r["sigma2"] if r["sigma2"] is not None else "",
                    int(r["switched"]) if r["switched"] is not None else "",
                    json.dumps(r["bbox"]) if r["bbox"] else "",
                ])

    mean_acc = float(np.mean(accs)) if accs else float("nan")
    logger.info("=" * 70)
    logger.info("modality=%s  Acc=%.4f  [%d sequences]",
                args.modality, mean_acc, len(accs))
    logger.info("Switches DETECT->TRACK : %d", n_switch_to_track)
    logger.info("Switches TRACK->DETECT : %d", n_switch_to_detect)
    logger.info("Frames in TRACK  : %d", total_track_frames)
    logger.info("Frames in DETECT : %d", total_detect_frames)
    logger.info(
        "thresholds: cls_det=%.3f loc_det=%.3f cls_trk=%.3f loc_trk=%.3f",
        cfg.theta_cls_detect, cfg.theta_loc_detect,
        cfg.theta_cls_track,  cfg.theta_loc_track,
    )
    logger.info("=" * 70)

    summary = {
        "modality":           args.modality,
        "unc_agg":            cfg.unc_agg,
        "acc":                mean_acc,
        "n_sequences":        len(accs),
        "switches_to_track":  n_switch_to_track,
        "switches_to_detect": n_switch_to_detect,
        "frames_in_track":    total_track_frames,
        "frames_in_detect":   total_detect_frames,
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
