"""
Preprocessing: adapts the Anti-UAV RGBT train and validation splits to the
format expected by EDTC's antiuav.py dataset loader.

Run this on Snellius via sbatch (see preprocess_trainval.job).

What it does per sequence:
  1. Extracts frames from infrared.mp4  -> ir_frames/000001.jpg, ...
     and copies infrared.json           -> IR_label.json
  2. Extracts frames from visible.mp4   -> rgb_frames/000001.jpg, ...
     and copies visible.json            -> visible_label.json
  3. Writes <split>/list.txt with one sequence name per line

Usage:
  python scripts/preprocess_antiuav_trainval.py
  python scripts/preprocess_antiuav_trainval.py --splits train
  python scripts/preprocess_antiuav_trainval.py --splits validation
  python scripts/preprocess_antiuav_trainval.py --splits train validation
"""
import argparse
import os
import shutil
import cv2

DATASET_ROOT = '/home/ssuttorp/Anti-UAV-RGBT'


def preprocess_split(split_dir: str, split_name: str) -> None:
    sequences = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    print(f'\n=== {split_name}: {len(sequences)} sequences ===\n')

    for i, seq in enumerate(sequences, 1):
        seq_dir = os.path.join(split_dir, seq)

        # ------------------------------------------------------------------
        # 1. IR frames: extract into seq_dir/ir_frames/
        # ------------------------------------------------------------------
        ir_video     = os.path.join(seq_dir, 'infrared.mp4')
        ir_out_dir   = os.path.join(seq_dir, 'ir_frames')
        ir_label_src = os.path.join(seq_dir, 'infrared.json')
        ir_label_dst = os.path.join(seq_dir, 'IR_label.json')

        first_ir_frame = os.path.join(ir_out_dir, '000001.jpg')
        if os.path.exists(first_ir_frame):
            print(f'[{i:03d}/{len(sequences)}] {seq} — IR: already done, skipping')
        elif not os.path.exists(ir_video):
            print(f'[{i:03d}/{len(sequences)}] WARNING: {seq} — infrared.mp4 not found, skipping IR')
        else:
            os.makedirs(ir_out_dir, exist_ok=True)
            cap = cv2.VideoCapture(ir_video)
            frame_idx = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(ir_out_dir, f'{frame_idx:06d}.jpg'), frame)
                frame_idx += 1
            cap.release()
            print(f'[{i:03d}/{len(sequences)}] {seq} — IR: extracted {frame_idx - 1} frames')

        if os.path.exists(ir_label_src) and not os.path.exists(ir_label_dst):
            shutil.copy2(ir_label_src, ir_label_dst)

        # ------------------------------------------------------------------
        # 2. RGB frames: extract into seq_dir/rgb_frames/
        # ------------------------------------------------------------------
        rgb_video     = os.path.join(seq_dir, 'visible.mp4')
        rgb_out_dir   = os.path.join(seq_dir, 'rgb_frames')
        rgb_label_src = os.path.join(seq_dir, 'visible.json')
        rgb_label_dst = os.path.join(seq_dir, 'visible_label.json')

        first_rgb_frame = os.path.join(rgb_out_dir, '000001.jpg')
        if os.path.exists(first_rgb_frame):
            print(f'[{i:03d}/{len(sequences)}] {seq} — RGB: already done, skipping')
        elif not os.path.exists(rgb_video):
            print(f'[{i:03d}/{len(sequences)}] WARNING: {seq} — visible.mp4 not found, skipping RGB')
        else:
            os.makedirs(rgb_out_dir, exist_ok=True)
            cap = cv2.VideoCapture(rgb_video)
            frame_idx = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(rgb_out_dir, f'{frame_idx:06d}.jpg'), frame)
                frame_idx += 1
            cap.release()
            print(f'[{i:03d}/{len(sequences)}] {seq} — RGB: extracted {frame_idx - 1} frames')

        if os.path.exists(rgb_label_src) and not os.path.exists(rgb_label_dst):
            shutil.copy2(rgb_label_src, rgb_label_dst)

    # ------------------------------------------------------------------
    # 3. Write list.txt
    # ------------------------------------------------------------------
    list_path = os.path.join(split_dir, 'list.txt')
    with open(list_path, 'w') as f:
        f.write('\n'.join(sequences))

    print(f'\nDone with {split_name}. Written {list_path} ({len(sequences)} sequences).')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        choices=['train', 'val'],
        help='Which splits to preprocess (default: train val)',
    )
    args = parser.parse_args()

    for split in args.splits:
        split_dir = os.path.join(DATASET_ROOT, split)
        if not os.path.isdir(split_dir):
            print(f'WARNING: {split_dir} does not exist, skipping {split}')
            continue
        preprocess_split(split_dir, split)

    print('\nAll done.')


if __name__ == '__main__':
    main()
