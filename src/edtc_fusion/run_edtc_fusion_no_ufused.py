"""
run_edtc_fusion_no_ufused.py — Ablation: drop the u_fused gate
==============================================================
Same pipeline as ``run_edtc_fusion`` (YOLO + SiamCAR, DBF/ABF fusion, DMS),
but the switching condition no longer uses fused vacuity u_fused.

Original gate (DETECT->TRACK, and TRACK maintenance):

    argmax(b_fused) == UAV  AND  u_fused < theta_cls  AND  sigma2_best < theta_loc

Ablation gate (this script):

    argmax(b_fused) == UAV  AND  sigma2_best < theta_loc

Implementation: we reuse the unmodified tracker and force the two u_fused
thresholds to +inf, so the u_fused < theta_cls check always passes.  The
remaining argmax and sigma2_best checks operate normally.

This isolates the contribution of u_fused-based gating from the rest of the
EDTC framework (b_fused argmax + sigma2_best gate + DMS routing).
"""

from __future__ import absolute_import, annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.edtc_fusion import run_edtc_fusion as _base


def main():
    args = _base.parse_args()
    # Disable the u_fused gates by making them vacuous (always pass).
    args.theta_cls_detect = math.inf
    args.theta_cls_track  = math.inf
    _run(args)


def _run(args):
    """Inline the body of _base.main() but with the parsed-args we patched.

    We can't call _base.main() directly because it re-runs parse_args().
    """
    import os
    import json
    import logging
    import numpy as np

    from src.edtc_fusion.tracker import EDTCFusionTracker, FusionConfig

    logger = logging.getLogger("edtc_fusion_no_ufused")

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
        theta_cls_detect=args.theta_cls_detect,   # +inf — ablation
        theta_loc_detect=args.theta_loc_detect,
        theta_cls_track=args.theta_cls_track,     # +inf — ablation
        theta_loc_track=args.theta_loc_track,
        yolo_imgsz=(args.yolo_imgsz, args.yolo_imgsz),
        yolo_conf_thres=args.yolo_conf,
        yolo_iou_thres=args.yolo_iou,
        yolo_classes=args.yolo_classes,
        device=args.device,
    )

    logger.info("Building EDTCFusionTracker (no-u_fused ablation)...")
    logger.info("theta_cls_detect=%s theta_cls_track=%s (u_fused gate disabled)",
                cfg.theta_cls_detect, cfg.theta_cls_track)
    tracker = EDTCFusionTracker(cfg)

    sequences = _base.discover_sequences(args.dataset_root)
    if args.sequence is not None:
        sequences = [s for s in sequences if s == args.sequence]
        if not sequences:
            raise SystemExit(f'Sequence "{args.sequence}" not found.')

    os.makedirs(args.results_dir, exist_ok=True)

    ir_accs, dms_accs = [], []
    n_switch_to_track = n_switch_to_detect = 0
    total_track_frames = total_detect_frames = 0

    for v_idx, seq_name in enumerate(sequences):
        seq_dir = os.path.join(args.dataset_root, seq_name)
        frames  = _base.load_paired_sequence(seq_dir)
        if not frames:
            logger.warning("No paired frames for %s — skipping.", seq_name)
            continue

        preds = _base.run_sequence(tracker, frames)

        seq_log     = tracker.frame_log
        ir_preds    = [r["ir_bbox"] if r.get("ir_bbox") else [0.0, 0.0, 0.0, 0.0]
                       for r in seq_log]
        dms_winners = [r.get("dms_winner") or "ir" for r in seq_log]

        acc_ir = _base.compute_edtc_acc(ir_preds, frames,
                                        alpha=args.alpha, beta=args.beta,
                                        failure_iou_thresh=args.failure_iou_thresh)
        acc_dms = _base.compute_edtc_acc_dms(preds, frames, dms_winners,
                                             alpha=args.alpha, beta=args.beta,
                                             failure_iou_thresh=args.failure_iou_thresh)
        if not math.isnan(acc_ir):  ir_accs.append(acc_ir)
        if not math.isnan(acc_dms): dms_accs.append(acc_dms)

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

        import csv
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
    logger.info("ABLATION: no u_fused gate (only b_fused argmax + sigma2_best < theta_loc)")
    logger.info("Acc_IR  = %.4f  [%d sequences]", mean_acc_ir,  len(ir_accs))
    logger.info("Acc_DMS = %.4f  [%d sequences]", mean_acc_dms, len(dms_accs))
    logger.info("Switches DETECT->TRACK : %d", n_switch_to_track)
    logger.info("Switches TRACK->DETECT : %d", n_switch_to_detect)
    logger.info("Frames in TRACK  : %d", total_track_frames)
    logger.info("Frames in DETECT : %d", total_detect_frames)
    logger.info("theta_loc_detect=%.3f theta_loc_track=%.3f (theta_cls_* = +inf, disabled)",
                cfg.theta_loc_detect, cfg.theta_loc_track)
    logger.info("=" * 70)

    summary = {
        "ablation":            "no_u_fused",
        "acc_ir":              mean_acc_ir,
        "acc_dms":             mean_acc_dms,
        "n_sequences":         len(ir_accs),
        "switches_to_track":   n_switch_to_track,
        "switches_to_detect":  n_switch_to_detect,
        "frames_in_track":     total_track_frames,
        "frames_in_detect":    total_detect_frames,
        "thresholds": {
            "theta_cls_detect": "inf (disabled)",
            "theta_loc_detect": cfg.theta_loc_detect,
            "theta_cls_track":  "inf (disabled)",
            "theta_loc_track":  cfg.theta_loc_track,
        },
        "fusion": cfg.fusion,
        "lam":    cfg.lam,
        "alpha":  args.alpha,
        "beta":   args.beta,
        "failure_iou_thresh": args.failure_iou_thresh,
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary -> %s", os.path.join(args.results_dir, "summary.json"))


if __name__ == "__main__":
    main()
