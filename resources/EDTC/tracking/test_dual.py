"""
test_dual.py — evaluation runner for the dual-stream tracker (Phase 3.1).

Runs DualStreamTracker on the Anti-UAV RGB-T paired test set.  Both YOLO
heads operate independently; no fusion is applied.

Results are written to two separate subdirectories so they can be evaluated
against the existing IR and RGB evaluation scripts independently:

  tracking_results/uavtrack_dual/<param>/          ← IR tracker boxes (target_bbox)
  tracking_results/uavtrack_dual/<param>_rgb/      ← RGB YOLO boxes

Usage (on Snellius):
  python tracking/test_dual.py \\
      --tracker_param baseline_dual \\
      --params__model /path/to/UAVTrackEH.pth.tar \\
      --params__evidential_threshold 0.2

The --tracker_param value must match a YAML filename under
experiments/uavtrack_dual/ (without the .yaml extension).
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.datasets import load_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.environment import env_settings


# ---------------------------------------------------------------------------
# Result I/O helpers
# ---------------------------------------------------------------------------

def _save_boxes(results_dir, seq_name, boxes):
    """Save a list of [x, y, w, h] boxes as a tab-separated text file."""
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, '%s.txt' % seq_name)
    arr = np.array(boxes, dtype=int)
    np.savetxt(out_path, arr, delimiter='\t', fmt='%d')


def _save_scores(results_dir, seq_name, scores):
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, '%s_score.txt' % seq_name)
    np.savetxt(out_path, np.array(scores, dtype=float), fmt='%.4f')


def _save_uncertainties(results_dir, seq_name, uncertainties):
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, '%s_uncertainty.txt' % seq_name)
    np.savetxt(out_path, np.array(uncertainties, dtype=float), fmt='%.4f')


# ---------------------------------------------------------------------------
# Per-sequence dual tracking loop
# ---------------------------------------------------------------------------

def run_dual_sequence(tracker_obj, seq, ir_results_dir, rgb_results_dir, debug=False):
    """Run dual-stream detection+tracking on one PairedSequence.

    Args:
        tracker_obj: lib.test.evaluation.tracker.Tracker instance.
        seq:         PairedSequence with .ir_frames and .rgb_frames lists.
        ir_results_dir:  directory to save IR tracker results.
        rgb_results_dir: directory to save RGB YOLO results.
        debug:       print per-frame timing if True.
    """
    params = tracker_obj.params
    tracker = tracker_obj.create_tracker(params)

    # Load both YOLO heads
    yolo_ir, yolo_rgb = tracker.initialize_yolo_dual()

    ir_boxes, rgb_boxes, scores, uncertainties = [], [], [], []

    # Use the shorter of the two frame lists to avoid index errors
    n_frames = min(len(seq.ir_frames), len(seq.rgb_frames))

    detection_flag = True  # start in detection mode

    def read_image(path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for frame_idx in range(n_frames):
        ir_img = read_image(seq.ir_frames[frame_idx])
        rgb_img = read_image(seq.rgb_frames[frame_idx])

        t0 = time.time()

        if detection_flag:
            out = tracker.initialize(ir_img, rgb_img, yolo_ir, yolo_rgb)
        else:
            out = tracker.track(ir_img, rgb_img)

        elapsed = time.time() - t0
        detection_flag = out['detection_flag']

        ir_boxes.append(out['ir_bbox'])
        rgb_boxes.append(out['rgb_bbox'])
        scores.append(out.get('uav_score', float('nan')))
        uncertainties.append(out.get('uav_uncertainty', float('nan')))

        if debug:
            print(f'  [{frame_idx+1:04d}/{n_frames}] '
                  f'det={detection_flag} '
                  f'ir={[round(v) for v in out["ir_bbox"]]} '
                  f'rgb={[round(v) for v in out["rgb_bbox"]]} '
                  f't={elapsed*1000:.1f}ms')

    _save_boxes(ir_results_dir, seq.name, ir_boxes)
    _save_boxes(rgb_results_dir, seq.name, rgb_boxes)
    _save_scores(ir_results_dir, seq.name, scores)
    _save_uncertainties(ir_results_dir, seq.name, uncertainties)

    print(f'  Saved {n_frames} frames → {ir_results_dir}/{seq.name}.txt')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run dual-stream UAV tracker on Anti-UAV RGB-T test set.'
    )
    parser.add_argument('--tracker_name', default='uavtrack_dual', type=str,
                        help='Tracker name (must match lib/test/tracker/<name>.py).')
    parser.add_argument('--tracker_param', default='baseline_dual', type=str,
                        help='Config file name under experiments/uavtrack_dual/ (no .yaml).')
    parser.add_argument('--dataset_name', default='antiuavrgbt', type=str,
                        help='Dataset name; must be "antiuavrgbt" for dual-stream.')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Run a single named sequence instead of the full test set.')
    parser.add_argument('--debug', type=int, default=0,
                        help='Debug level: 0=off, 1=per-frame print.')
    parser.add_argument('--params__model', type=str, required=True,
                        help='Path to UAVTrackEH checkpoint (.pth.tar).')
    parser.add_argument('--params__evidential_threshold', type=float, default=0.2,
                        help='Uncertainty threshold for re-detection.')
    parser.add_argument('--params__search_area_scale', type=float, default=4.55)
    parser.add_argument('--params__update_interval', type=int, default=None)
    parser.add_argument('--params__online_sizes', type=int, default=None)
    parser.add_argument('--params__max_score_decay', type=float, default=1.0)

    args = parser.parse_args()

    # Build tracker_params dict from --params__* arguments
    tracker_params = {}
    for attr in dir(args):
        parts = attr.split('__')
        if parts[0] == 'params' and getattr(args, attr) is not None:
            tracker_params[parts[1]] = getattr(args, attr)
    print('tracker_params:', tracker_params)

    # Construct Tracker (loads config + parameters)
    tracker_obj = Tracker(
        args.tracker_name, args.tracker_param, args.dataset_name,
        run_id=None, tracker_params=tracker_params,
    )

    # Result directories
    env = env_settings()
    ir_results_dir = os.path.join(
        env.results_path, args.tracker_name, args.tracker_param
    )
    rgb_results_dir = os.path.join(
        env.results_path, args.tracker_name, args.tracker_param + '_rgb'
    )

    # Load paired dataset
    sequences = load_dataset(args.dataset_name)
    if args.sequence is not None:
        sequences = [s for s in sequences if s.name == args.sequence]
        if not sequences:
            raise ValueError(f'Sequence "{args.sequence}" not found in dataset.')

    total = len(sequences)
    for idx, seq in enumerate(sequences, start=1):
        print(f'[{idx:03d}/{total:03d}] {seq.name}')
        run_dual_sequence(
            tracker_obj, seq,
            ir_results_dir, rgb_results_dir,
            debug=(args.debug > 0),
        )

    print(f'\nDone. IR results  → {ir_results_dir}')
    print(f'      RGB results → {rgb_results_dir}')


if __name__ == '__main__':
    main()
