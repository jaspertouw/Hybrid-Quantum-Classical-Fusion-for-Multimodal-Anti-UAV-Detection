"""
Diagnostic script: compare per-sequence frame counts between IR and RGB.

Checks for each sequence in list.txt:
  - Number of frames in IR_label.json vs visible_label.json
  - Which result files (.txt) are present / missing in each results folder

Usage (local):
    python scripts/check_frame_counts.py \
        --dataset_dir "C:/Users/shara/Documents/MasterIS/thesis/Anti-UAV-RGBT/test"

Usage (Snellius, with results folders):
    python scripts/check_frame_counts.py \
        --dataset_dir /home/ssuttorp/Anti-UAV-RGBT/test \
        --ir_results  /home/ssuttorp/mm-uav-threat-detection/resources/EDTC/test/tracking_results/uavtrack_eh/baseline \
        --rgb_results /home/ssuttorp/mm-uav-threat-detection/resources/EDTC/test/tracking_results/uavtrack_eh/baseline_rgb
"""

import argparse
import json
import os


def load_label(seq_path, filename):
    path = os.path.join(seq_path, filename)
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        label = json.load(f)
    return label


def count_result_frames(results_dir, seq_name):
    """Count lines in {seq}.txt if it exists, else return None."""
    if results_dir is None:
        return None
    path = os.path.join(results_dir, seq_name + '.txt')
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True,
                        help='Dataset split dir containing list.txt and sequence folders.')
    parser.add_argument('--ir_results', default=None,
                        help='Optional: IR tracking results dir to check for missing .txt files.')
    parser.add_argument('--rgb_results', default=None,
                        help='Optional: RGB tracking results dir to check for missing .txt files.')
    args = parser.parse_args()

    list_file = os.path.join(args.dataset_dir, 'list.txt')
    with open(list_file, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]

    print(f"Total sequences in list.txt: {len(sequences)}\n")

    mismatches      = []
    missing_ir_gt   = []
    missing_rgb_gt  = []
    missing_ir_res  = []
    missing_rgb_res = []

    total_ir_frames  = 0
    total_rgb_frames = 0

    for seq in sequences:
        seq_path = os.path.join(args.dataset_dir, seq)

        ir_label  = load_label(seq_path, 'IR_label.json')
        rgb_label = load_label(seq_path, 'visible_label.json')

        ir_n  = len(ir_label['gt_rect'])  if ir_label  else None
        rgb_n = len(rgb_label['gt_rect']) if rgb_label else None

        if ir_label  is None: missing_ir_gt.append(seq)
        if rgb_label is None: missing_rgb_gt.append(seq)

        if ir_n is not None:  total_ir_frames  += ir_n
        if rgb_n is not None: total_rgb_frames += rgb_n

        if ir_n is not None and rgb_n is not None and ir_n != rgb_n:
            mismatches.append((seq, ir_n, rgb_n))

        ir_res_n  = count_result_frames(args.ir_results,  seq)
        rgb_res_n = count_result_frames(args.rgb_results, seq)

        if args.ir_results  is not None and ir_res_n  is None: missing_ir_res.append(seq)
        if args.rgb_results is not None and rgb_res_n is None: missing_rgb_res.append(seq)

    # Summary
    print(f"{'='*60}")
    print(f"  GT frame counts (from label files)")
    print(f"{'='*60}")
    print(f"  Total IR  frames (IR_label.json):       {total_ir_frames:>8}")
    print(f"  Total RGB frames (visible_label.json):  {total_rgb_frames:>8}")
    print(f"  Difference:                             {abs(total_ir_frames - total_rgb_frames):>8}")

    if mismatches:
        print(f"\n  Sequences where IR != RGB frame count ({len(mismatches)}):")
        print(f"  {'Sequence':<35} {'IR':>6} {'RGB':>6} {'diff':>6}")
        print(f"  {'-'*55}")
        for seq, ir_n, rgb_n in sorted(mismatches, key=lambda x: abs(x[1]-x[2]), reverse=True):
            print(f"  {seq:<35} {ir_n:>6} {rgb_n:>6} {ir_n-rgb_n:>+6}")
    else:
        print(f"\n  All sequences: IR and RGB frame counts match.")

    if missing_ir_gt:
        print(f"\n  Missing IR_label.json ({len(missing_ir_gt)}): {missing_ir_gt}")
    if missing_rgb_gt:
        print(f"\n  Missing visible_label.json ({len(missing_rgb_gt)}): {missing_rgb_gt}")

    if args.ir_results is not None:
        print(f"\n{'='*60}")
        print(f"  Tracking results presence")
        print(f"{'='*60}")
        ir_found  = len(sequences) - len(missing_ir_res)
        rgb_found = len(sequences) - len(missing_rgb_res)
        print(f"  IR  results:  {ir_found}/{len(sequences)} sequences found")
        print(f"  RGB results:  {rgb_found}/{len(sequences)} sequences found")

        if missing_ir_res:
            print(f"\n  Missing IR result files ({len(missing_ir_res)}):")
            for s in missing_ir_res: print(f"    {s}")
        if missing_rgb_res:
            print(f"\n  Missing RGB result files ({len(missing_rgb_res)}):")
            for s in missing_rgb_res: print(f"    {s}")

        # Per-sequence line count vs GT
        print(f"\n{'='*60}")
        print(f"  Per-sequence frame count: predictions vs GT")
        print(f"{'='*60}")

        total_ir_pred  = 0
        total_rgb_pred = 0
        short_ir  = []
        short_rgb = []

        for seq in sequences:
            seq_path  = os.path.join(args.dataset_dir, seq)
            ir_label  = load_label(seq_path, 'IR_label.json')
            gt_n      = len(ir_label['gt_rect']) if ir_label else 0

            ir_res_n  = count_result_frames(args.ir_results,  seq)
            rgb_res_n = count_result_frames(args.rgb_results, seq)

            if ir_res_n  is not None: total_ir_pred  += ir_res_n
            if rgb_res_n is not None: total_rgb_pred += rgb_res_n

            if ir_res_n  is not None and ir_res_n  < gt_n:
                short_ir.append((seq, gt_n, ir_res_n))
            if rgb_res_n is not None and rgb_res_n < gt_n:
                short_rgb.append((seq, gt_n, rgb_res_n))

        print(f"  Total IR  prediction frames:  {total_ir_pred:>8}  (GT: {total_ir_frames})")
        print(f"  Total RGB prediction frames:  {total_rgb_pred:>8}  (GT: {total_rgb_frames})")

        if short_ir:
            print(f"\n  IR sequences shorter than GT ({len(short_ir)}):")
            print(f"  {'Sequence':<35} {'GT':>6} {'pred':>6} {'missing':>8}")
            print(f"  {'-'*57}")
            for seq, gt_n, pred_n in sorted(short_ir, key=lambda x: x[1]-x[2], reverse=True):
                print(f"  {seq:<35} {gt_n:>6} {pred_n:>6} {gt_n-pred_n:>8}")
        else:
            print(f"\n  IR: all sequences have full prediction coverage.")

        if short_rgb:
            print(f"\n  RGB sequences shorter than GT ({len(short_rgb)}):")
            print(f"  {'Sequence':<35} {'GT':>6} {'pred':>6} {'missing':>8}")
            print(f"  {'-'*57}")
            for seq, gt_n, pred_n in sorted(short_rgb, key=lambda x: x[1]-x[2], reverse=True):
                print(f"  {seq:<35} {gt_n:>6} {pred_n:>6} {gt_n-pred_n:>8}")
        else:
            print(f"\n  RGB: all sequences have full prediction coverage.")


if __name__ == '__main__':
    main()
