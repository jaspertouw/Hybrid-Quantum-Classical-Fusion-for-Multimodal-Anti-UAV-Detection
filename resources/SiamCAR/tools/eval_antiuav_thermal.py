"""
Track every sequence in the Anti-UAV thermal validation split with the
fine-tuned SiamCAR checkpoint and compute EDTC Acc at two IoU thresholds.

EDTC Acc(threshold) per sequence:
    acc = (#frames where IoU(pred, gt) >= threshold AND target exists)
          / (#frames where target exists)

Overall Acc = mean(acc) across all sequences.

Usage (from resources/SiamCAR/tools/):
    python eval_antiuav_thermal.py \
        --snapshot  /home/ssuttorp/pretrained_models/siamcar_thermal/checkpoint_e20.pth \
        --config    ../experiments/siamcar_r50/config_antiuav_thermal.yaml \
        --dataset_root /home/ssuttorp/Anti-UAV-RGBT/val \
        --alpha 0.2 --beta 0.3 \
        --results_dir /home/ssuttorp/pretrained_models/siamcar_thermal/val_results
"""

from __future__ import absolute_import
import argparse
import json
import logging
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamcar_tracker import SiamCARTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_sequence(seq_dir):
    """Return list of (frame_path, [x,y,w,h] | None) ordered by frame index."""
    label_path = os.path.join(seq_dir, 'IR_label.json')
    with open(label_path) as f:
        ann = json.load(f)

    exist = ann['exist']
    gt_rect = ann['gt_rect']
    frames_dir = os.path.join(seq_dir, 'ir_frames')

    data = []
    for i, (ex, rect) in enumerate(zip(exist, gt_rect)):
        frame_path = os.path.join(frames_dir, '{:06d}.jpg'.format(i + 1))
        data.append((frame_path, rect if ex else None))
    return data


def iou_xywh(pred, gt):
    """IoU between two [x, y, w, h] bboxes."""
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    ix1 = max(px, gx)
    iy1 = max(py, gy)
    ix2 = min(px + pw, gx + gw)
    iy2 = min(py + ph, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return inter / (union + 1e-8)


def track_sequence(tracker, frames, hp):
    """Run tracker; return list of predicted [x, y, w, h] bboxes."""
    initialized = False
    preds = []

    for idx, (frame_path, gt_bbox) in enumerate(frames):
        img = cv2.imread(frame_path)
        if img is None:
            logger.warning("cannot read %s", frame_path)
            preds.append([0.0, 0.0, 0.0, 0.0])
            continue

        if not initialized:
            if gt_bbox is None:
                # target absent — store dummy, try next frame
                preds.append([0.0, 0.0, 0.0, 0.0])
                continue
            if gt_bbox[2] <= 0 or gt_bbox[3] <= 0:
                # exist=1 but degenerate bbox (w=0 or h=0) — skip frame
                preds.append([0.0, 0.0, 0.0, 0.0])
                continue
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
            init_bbox = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
            tracker.init(img, init_bbox)
            preds.append(init_bbox)
            initialized = True
        else:
            outputs = tracker.track(img, hp)
            preds.append(outputs['bbox'])

    return preds


def compute_edtc_acc(preds, frames, alpha=0.2, beta=0.3, failure_iou_thresh=0.2):
    """
    EDTC Acc metric (equation from the EDTC paper):

        Acc = [ Σ_t  IoU_t·δ(v_t>0) + p_t·(1−δ(v_t>0)) ] / T
              − α · ( Σ_{t: v_t>0} q_t / T* ) ^ β

    Terms:
        IoU_t   — IoU between pred and gt at frame t
        v_t     — GT visibility (1=present, 0=absent)
        p_t     — 1 if system predicts absent (empty/zero box), else 0
        q_t     — 1 if localization fails when present (IoU_t < failure_iou_thresh)
        T       — total frames
        T*      — frames where target is present
        α=0.2, β=0.3 — penalty weight and exponent (EDTC constants)
        failure_iou_thresh=0.2 — IoU below which a prediction counts as failure

    Note: a pure tracker never predicts absent (p_t always 0), so absent frames
    always contribute 0 to the first term. This is the correct behaviour —
    absence detection requires the full EDTC pipeline.
    """
    T = len(frames)
    if T == 0:
        return float('nan')

    T_star = sum(1 for _, gt in frames if gt is not None)

    reward_sum = 0.0
    failure_sum = 0.0

    for pred, (_, gt_bbox) in zip(preds, frames):
        target_present = gt_bbox is not None

        # p_t: system predicts absent when pred box is zero/degenerate
        pred_empty = (pred[2] <= 0 or pred[3] <= 0)
        p_t = 1.0 if pred_empty else 0.0

        if target_present:
            iou_t = iou_xywh(pred, gt_bbox) if not pred_empty else 0.0
            reward_sum += iou_t
            # q_t: localization failure (IoU below threshold)
            q_t = 1.0 if iou_t < failure_iou_thresh else 0.0
            failure_sum += q_t
        else:
            # target absent: reward only if system correctly predicts absent
            reward_sum += p_t

    first_term = reward_sum / T
    penalty = alpha * ((failure_sum / T_star) ** beta) if T_star > 0 else 0.0

    return first_term - penalty


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description='Anti-UAV Thermal — SiamCAR eval')
    parser.add_argument('--snapshot', required=True,
                        help='path to checkpoint (.pth)')
    parser.add_argument('--config',
                        default='../experiments/siamcar_r50/config_antiuav_thermal.yaml',
                        help='SiamCAR config yaml')
    parser.add_argument('--dataset_root', required=True,
                        help='Anti-UAV val split root (contains sequence dirs + list.txt)')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='EDTC penalty weight (default 0.2, fixed in EDTC paper)')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='EDTC penalty exponent (default 0.3, fixed in EDTC paper)')
    parser.add_argument('--failure_iou_thresh', type=float, default=0.2,
                        help='IoU below which a prediction counts as localization failure (q_t=1)')
    parser.add_argument('--results_dir', default='',
                        help='if set, save per-sequence bbox predictions here')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    model = ModelBuilder()
    model = load_pretrain(model, args.snapshot).cuda().eval()
    tracker = SiamCARTracker(model, cfg.TRACK)

    hp = {
        'lr':        cfg.HP_SEARCH.ANTIUAVTHERMAL[0],
        'penalty_k': cfg.HP_SEARCH.ANTIUAVTHERMAL[1],
        'window_lr': cfg.HP_SEARCH.ANTIUAVTHERMAL[2],
    }

    # Discover sequences
    list_file = os.path.join(args.dataset_root, 'list.txt')
    if os.path.exists(list_file):
        with open(list_file) as f:
            sequences = [s.strip() for s in f if s.strip()]
    else:
        sequences = sorted(
            d for d in os.listdir(args.dataset_root)
            if os.path.isdir(os.path.join(args.dataset_root, d))
        )

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

    accs = []

    for v_idx, seq_name in enumerate(sequences):
        seq_dir = os.path.join(args.dataset_root, seq_name)
        frames = load_sequence(seq_dir)
        preds = track_sequence(tracker, frames, hp)

        acc = compute_edtc_acc(preds, frames,
                               alpha=args.alpha,
                               beta=args.beta,
                               failure_iou_thresh=args.failure_iou_thresh)

        if not math.isnan(acc):
            accs.append(acc)

        logger.info(
            '[%3d/%3d] %-30s  Acc=%.4f',
            v_idx + 1, len(sequences), seq_name, acc,
        )

        if args.results_dir:
            out_path = os.path.join(args.results_dir, seq_name + '.txt')
            with open(out_path, 'w') as f:
                for bbox in preds:
                    f.write(','.join('{:.4f}'.format(v) for v in bbox) + '\n')

    mean_acc = float(np.mean(accs)) if accs else float('nan')

    logger.info('=' * 60)
    logger.info('EDTC Acc = %.4f  [%d sequences]', mean_acc, len(accs))
    logger.info('  alpha=%.1f (penalty weight)  beta=%.1f (penalty exponent)  '
                'failure_iou_thresh=%.2f', args.alpha, args.beta, args.failure_iou_thresh)
    logger.info('Checkpoint: %s', args.snapshot)
    logger.info('=' * 60)

    if args.results_dir:
        summary = {
            'checkpoint':          args.snapshot,
            'alpha':               args.alpha,
            'beta':                args.beta,
            'failure_iou_thresh':  args.failure_iou_thresh,
            'acc':                 mean_acc,
            'n_sequences':         len(sequences),
        }
        summary_path = os.path.join(args.results_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info('Summary -> %s', summary_path)


if __name__ == '__main__':
    main()
