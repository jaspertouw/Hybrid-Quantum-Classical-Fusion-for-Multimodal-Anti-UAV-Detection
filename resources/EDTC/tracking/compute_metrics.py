"""
Post-processing evaluation script for UAV tracking experiments.

Reads saved tracker outputs and ground truth, then computes:
  Accuracy, Mean IoU, ECE, AUROC, AUSE (sparsification), FPS

for two confidence signals:
  - conf_prob:  prob[0][1]  (Dirichlet expected P(target))
  - conf_1mu:   1 - uncertainty = 1 - 2/sum(alpha)

Usage:
    python tracking/compute_metrics.py \
        --results_dir /path/to/tracking_results/uavtrack_eh/baseline \
        --dataset_dir /path/to/Anti-UAV-RGBT/val \
        --iou_threshold 0.5

Outputs a summary table to stdout and saves metrics_summary.json in results_dir.
"""

import argparse
import json
import os
import sys

import numpy as np

# Allow imports from the EDTC package root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.utils.eval_metrics import (
    iou_xywh,
    compute_accuracy,
    compute_mean_iou,
    compute_ece,
    compute_auroc,
    compute_sparsification,
    compute_fps,
)


def load_predictions(results_dir, seq_name):
    """Load bbox, conf_prob, conf_uncertainty, and time files for one sequence."""
    base = os.path.join(results_dir, seq_name)

    bbox_file = base + '.txt'
    with open(bbox_file, 'r') as f:
        rows = [line.strip().split('\t') for line in f if line.strip()]
    bboxes = [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in rows]

    def _load_floats(path):
        if not os.path.isfile(path):
            return None
        with open(path, 'r') as f:
            return [float(line.strip()) for line in f if line.strip()]

    conf_prob = _load_floats(base + '_conf_prob.txt')
    conf_unc  = _load_floats(base + '_conf_uncertainty.txt')
    times     = _load_floats(base + '_time.txt')

    return bboxes, conf_prob, conf_unc, times


def load_groundtruth(dataset_dir, seq_name, label_file='IR_label.json'):
    """Load GT label file for one sequence. Returns (gt_rects, exists)."""
    path = os.path.join(dataset_dir, seq_name, label_file)
    with open(path, 'r') as f:
        label = json.load(f)
    return label['gt_rect'], label['exist']


def build_per_frame_arrays(bboxes, conf_prob, conf_unc, gt_rects, exists, iou_threshold):
    """
    Build aligned per-frame arrays for metric computation.

    Frames where conf_prob or conf_unc is NaN are init/detection frames (YOLO ran,
    not the tracker). These frames still contribute to Accuracy and IoU, but are
    excluded from confidence-based metrics (ECE, AUROC, sparsification).

    Returns:
        correctness:      (N,) int   — 1 if prediction correct, else 0 (all frames)
        ious_present:     (M,) float — IoU on exist=True frames only
        conf_prob_all:    (K,) float — prob confidence, NaN frames excluded
        conf_1mu_all:     (K,) float — 1-unc confidence, NaN frames excluded
        correctness_conf: (K,) int   — correctness aligned to conf_prob_all/conf_1mu_all
        conf_prob_pres:   (L,) float — prob confidence, exist=True and non-NaN frames
        conf_1mu_pres:    (L,) float — 1-unc confidence, exist=True and non-NaN frames
        ious_conf_pres:   (L,) float — IoU aligned to conf_prob_pres/conf_1mu_pres
    """
    n = min(len(bboxes), len(gt_rects), len(exists))

    correctness      = []
    ious_present     = []
    cp_all, cm_all   = [], []
    correct_conf     = []
    cp_pres, cm_pres = [], []
    ious_conf_pres   = []

    for i in range(n):
        pred  = bboxes[i]
        gt    = gt_rects[i]
        exist = exists[i]
        cp    = conf_prob[i] if conf_prob is not None else float('nan')
        unc   = conf_unc[i]  if conf_unc  is not None else float('nan')
        cm    = 1.0 - unc

        cp_nan = (cp != cp)  # True if NaN
        cm_nan = (cm != cm)

        if exist:
            frame_iou = iou_xywh(pred, gt)
            correct   = int(frame_iou >= iou_threshold)
            ious_present.append(frame_iou)
            if not cp_nan and not cm_nan:
                cp_pres.append(cp)
                cm_pres.append(cm)
                ious_conf_pres.append(frame_iou)
        else:
            # Correct on absent frame = model predicts background (conf_prob < 0.5)
            correct = int(cp < 0.5) if not cp_nan else 0

        correctness.append(correct)

        if not cp_nan and not cm_nan:
            cp_all.append(cp)
            cm_all.append(cm)
            correct_conf.append(correct)

    return (
        np.array(correctness),
        np.array(ious_present),
        np.array(cp_all),
        np.array(cm_all),
        np.array(correct_conf),
        np.array(cp_pres),
        np.array(cm_pres),
        np.array(ious_conf_pres),
    )


def run(results_dir, dataset_dir, iou_threshold, label_file='IR_label.json'):
    list_file = os.path.join(dataset_dir, 'list.txt')
    with open(list_file, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]

    # Per-sequence accumulators
    all_correctness    = []
    all_ious_present   = []
    all_cp_all, all_cm_all       = [], []
    all_correct_conf             = []
    all_cp_pres, all_cm_pres     = [], []
    all_ious_conf_pres           = []
    all_times = []

    skipped = []
    for seq in sequences:
        try:
            bboxes, conf_prob, conf_unc, times = load_predictions(results_dir, seq)
            gt_rects, exists                   = load_groundtruth(dataset_dir, seq, label_file)
        except FileNotFoundError as e:
            skipped.append(seq)
            continue

        correctness, ious_p, cp_all, cm_all, correct_conf, cp_p, cm_p, ious_cp = build_per_frame_arrays(
            bboxes, conf_prob, conf_unc, gt_rects, exists, iou_threshold
        )
        all_correctness.append(correctness)
        all_ious_present.append(ious_p)
        all_cp_all.append(cp_all)
        all_cm_all.append(cm_all)
        all_correct_conf.append(correct_conf)
        all_cp_pres.append(cp_p)
        all_cm_pres.append(cm_p)
        all_ious_conf_pres.append(ious_cp)
        if times is not None:
            all_times.extend(times)

    if not all_correctness:
        print("No sequences found. Check --results_dir and --dataset_dir.")
        return

    correctness_cat      = np.concatenate(all_correctness)
    ious_present_cat     = np.concatenate(all_ious_present)
    cp_all_cat           = np.concatenate(all_cp_all)
    cm_all_cat           = np.concatenate(all_cm_all)
    correct_conf_cat     = np.concatenate(all_correct_conf)
    cp_pres_cat          = np.concatenate(all_cp_pres)
    cm_pres_cat          = np.concatenate(all_cm_pres)
    ious_conf_pres_cat   = np.concatenate(all_ious_conf_pres)

    results = {}

    # Metrics that don't depend on confidence signal (use all frames)
    results['accuracy']       = compute_accuracy(correctness_cat)
    results['mean_iou']       = compute_mean_iou(ious_present_cat)
    results['fps']            = compute_fps(np.array(all_times)) if all_times else float('nan')
    results['n_frames']       = int(len(correctness_cat))
    results['n_present']      = int(len(ious_present_cat))
    results['n_conf_frames']  = int(len(cp_all_cat))  # frames with valid confidence (non-NaN)

    # Metrics per confidence signal (NaN init frames excluded)
    for label, confs_all, confs_pres, ious_pres in [
        ('prob', cp_all_cat, cp_pres_cat, ious_conf_pres_cat),
        ('1mu',  cm_all_cat, cm_pres_cat, ious_conf_pres_cat),
    ]:
        results[f'ece_{label}']   = compute_ece(confs_all, correct_conf_cat)
        results[f'auroc_{label}'] = compute_auroc(confs_all, correct_conf_cat)
        sp = compute_sparsification(ious_pres, confs_pres)
        results[f'ause_{label}']  = sp['ause']

    # Print summary
    print(f"\n{'='*62}")
    print(f"  Results: {results_dir}")
    print(f"  Frames: {results['n_frames']} total, {results['n_present']} present")
    print(f"  Conf frames (non-init): {results['n_conf_frames']}")
    print(f"  IoU threshold: {iou_threshold}")
    if skipped:
        print(f"  Skipped sequences ({len(skipped)}): {skipped[:5]}{'...' if len(skipped)>5 else ''}")
    print(f"{'='*62}")
    print(f"{'Metric':<22} {'prob-based':>12} {'1-unc-based':>12}")
    print(f"{'-'*46}")
    print(f"{'Accuracy':<22} {results['accuracy']:>12.4f} {'':>12}")
    print(f"{'Mean IoU (present)':<22} {results['mean_iou']:>12.4f} {'':>12}")
    print(f"{'ECE':<22} {results['ece_prob']:>12.4f} {results['ece_1mu']:>12.4f}")
    print(f"{'AUROC':<22} {results['auroc_prob']:>12.4f} {results['auroc_1mu']:>12.4f}")
    print(f"{'AUSE':<22} {results['ause_prob']:>12.4f} {results['ause_1mu']:>12.4f}")
    print(f"{'FPS':<22} {results['fps']:>12.1f} {'':>12}")
    print(f"{'='*62}\n")

    # Save JSON
    out_path = os.path.join(results_dir, 'metrics_summary.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute evaluation metrics from saved tracker outputs.')
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing {seq}.txt, {seq}_conf_prob.txt, etc.')
    parser.add_argument('--dataset_dir', required=True,
                        help='Dataset split directory containing list.txt and sequence folders.')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for correctness on present frames (default: 0.5).')
    parser.add_argument('--label_file', type=str, default='IR_label.json',
                        help='GT label filename inside each sequence folder (default: IR_label.json). '
                             'Use visible_label.json for RGB baselines.')
    args = parser.parse_args()
    run(args.results_dir, args.dataset_dir, args.iou_threshold, args.label_file)
