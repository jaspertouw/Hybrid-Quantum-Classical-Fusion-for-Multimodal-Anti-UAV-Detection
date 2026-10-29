"""
run_edtc_single_no_ufused.py — Ablation: drop the u gate (single modality)
==========================================================================
Same pipeline as ``run_edtc_single`` (one modality: YOLO + SiamCAR) but the
switching condition no longer uses the per-modality vacuity u.

Original gate (DETECT->TRACK, and TRACK maintenance):

    argmax(b) == UAV  AND  u < theta_cls  AND  sigma2 < theta_loc

Ablation gate:

    argmax(b) == UAV  AND  sigma2 < theta_loc

Implementation: reuse the unmodified ``EDTCSingleTracker`` and force the two
theta_cls thresholds to +inf so the u < theta_cls check always passes.  The
argmax(b) and sigma2 checks operate normally.

The same script handles --modality ir and --modality rgb.
"""

from __future__ import absolute_import, annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.edtc_fusion import run_edtc_single as _base


def main():
    args = _base.parse_args()
    # Disable the u gates by making them vacuous (always pass).
    args.theta_cls_detect = math.inf
    args.theta_cls_track  = math.inf
    _run(args)


def _run(args):
    """Body of _base.main() with patched args.  Can't call _base.main() directly
    because it re-runs parse_args()."""
    import os
    import csv
    import json
    import logging
    import numpy as np

    from src.edtc_fusion.tracker_single import EDTCSingleTracker, SingleConfig
    from src.edtc_fusion.run_edtc_fusion import (
        compute_edtc_acc,
        load_paired_sequence,
        discover_sequences,
    )

    logger = logging.getLogger("edtc_single_no_ufused")

    cfg = SingleConfig(
        modality=args.modality,
        yolo_weights=args.yolo,
        yolo_base=args.yolo_base,
        siam_stage2=args.siam_stage2,
        siam_stage1=args.siam_stage1,
        siam_pretrained=args.siam_pretrained,
        siam_cfg=args.siam_cfg,
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

    logger.info("Building EDTCSingleTracker (modality=%s, no-u ablation)...", args.modality)
    logger.info("theta_cls_detect=%s theta_cls_track=%s (u gate disabled)",
                cfg.theta_cls_detect, cfg.theta_cls_track)
    tracker = EDTCSingleTracker(cfg)

    sequences = discover_sequences(args.dataset_root)
    if args.sequence is not None:
        sequences = [s for s in sequences if s == args.sequence]
        if not sequences:
            raise SystemExit(f'Sequence "{args.sequence}" not found.')

    os.makedirs(args.results_dir, exist_ok=True)
    gt_key = "gt" if args.modality == "ir" else "rgb_gt"

    accs = []
    n_switch_to_track = n_switch_to_detect = 0
    total_track_frames = total_detect_frames = 0

    for v_idx, seq_name in enumerate(sequences):
        seq_dir = os.path.join(args.dataset_root, seq_name)
        frames  = load_paired_sequence(seq_dir)
        if not frames:
            logger.warning("No paired frames for %s — skipping.", seq_name)
            continue

        preds = _base.run_sequence(tracker, frames, args.modality)

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

        bbox_path = os.path.join(args.results_dir, seq_name + ".txt")
        with open(bbox_path, "w") as f:
            for bbox in preds:
                f.write(",".join("{:.4f}".format(v) for v in bbox) + "\n")

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
    logger.info("ABLATION: no u gate (only argmax(b)==UAV + sigma2 < theta_loc)")
    logger.info("modality=%s  Acc=%.4f  [%d sequences]",
                args.modality, mean_acc, len(accs))
    logger.info("Switches DETECT->TRACK : %d", n_switch_to_track)
    logger.info("Switches TRACK->DETECT : %d", n_switch_to_detect)
    logger.info("Frames in TRACK  : %d", total_track_frames)
    logger.info("Frames in DETECT : %d", total_detect_frames)
    logger.info("theta_loc_detect=%.3f theta_loc_track=%.3f (theta_cls_* = +inf, disabled)",
                cfg.theta_loc_detect, cfg.theta_loc_track)
    logger.info("=" * 70)

    summary = {
        "ablation":           "no_u",
        "modality":           args.modality,
        "acc":                mean_acc,
        "n_sequences":        len(accs),
        "switches_to_track":  n_switch_to_track,
        "switches_to_detect": n_switch_to_detect,
        "frames_in_track":    total_track_frames,
        "frames_in_detect":   total_detect_frames,
        "thresholds": {
            "theta_cls_detect": "inf (disabled)",
            "theta_loc_detect": cfg.theta_loc_detect,
            "theta_cls_track":  "inf (disabled)",
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
