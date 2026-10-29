"""
Convert Anti-UAV RGB (visible) JSON annotations to YOLO label format for YOLOv5 training.

Mirrors prepare_yolo_labels.py but for the visible modality.

Each sequence already has an images/ symlink pointing to ir_frames/ (from the IR
label prep). To avoid a naming conflict, RGB uses a rgb_yolo/ subdirectory:
  <seq_dir>/rgb_yolo/images  ->  ../rgb_frames   (symlink)
  <seq_dir>/rgb_yolo/labels/000001.txt            (YOLO label files)

YOLOv5 label resolution:
    .../rgb_yolo/images/000001.jpg  ->  .../rgb_yolo/labels/000001.txt  ✓

Usage (run on Snellius, see prepare_yolo_labels_rgb.job):
    python scripts/prepare_yolo_labels_rgb.py
    python scripts/prepare_yolo_labels_rgb.py --splits train
    python scripts/prepare_yolo_labels_rgb.py --splits train val
"""

import argparse
import json
import os

import cv2


DATASET_ROOT = '/home/ssuttorp/Anti-UAV-RGBT'


def get_image_size(rgb_frames_dir: str):
    """Return (width, height) by reading the first frame in rgb_frames_dir."""
    for fname in sorted(os.listdir(rgb_frames_dir)):
        if fname.lower().endswith('.jpg'):
            img = cv2.imread(os.path.join(rgb_frames_dir, fname))
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    raise RuntimeError(f'No readable JPEG found in {rgb_frames_dir}')


def process_sequence(seq_dir: str) -> list:
    """
    Write YOLO label files and create rgb_yolo/images symlink for one sequence.

    Returns a sorted list of absolute image paths via the rgb_yolo/images symlink,
    so YOLOv5's label resolver can substitute /images/ -> /labels/.
    """
    rgb_frames_dir = os.path.join(seq_dir, 'rgb_frames')
    rgb_yolo_dir   = os.path.join(seq_dir, 'rgb_yolo')
    images_link    = os.path.join(rgb_yolo_dir, 'images')   # symlink -> ../../rgb_frames
    labels_dir     = os.path.join(rgb_yolo_dir, 'labels')
    label_json     = os.path.join(seq_dir, 'visible_label.json')

    if not os.path.isdir(rgb_frames_dir):
        print(f'  WARNING: rgb_frames/ missing in {seq_dir}, skipping')
        return []
    if not os.path.isfile(label_json):
        print(f'  WARNING: visible_label.json missing in {seq_dir}, skipping')
        return []

    # --- 1. Create rgb_yolo/images symlink pointing to ../rgb_frames ----------
    os.makedirs(rgb_yolo_dir, exist_ok=True)
    if not os.path.exists(images_link):
        # Relative to rgb_yolo/, ../rgb_frames resolves to seq_dir/rgb_frames
        os.symlink('../rgb_frames', images_link)

    with open(label_json, 'r') as f:
        label = json.load(f)

    gt_rects = label['gt_rect']   # list of [x, y, w, h] in pixels (top-left origin)
    exists   = label['exist']     # list of 0/1

    img_w, img_h = get_image_size(rgb_frames_dir)

    # --- 2. Write YOLO label files into rgb_yolo/labels/ ---------------------
    os.makedirs(labels_dir, exist_ok=True)

    frame_files = sorted(
        f for f in os.listdir(rgb_frames_dir) if f.lower().endswith('.jpg')
    )
    n = min(len(frame_files), len(gt_rects), len(exists))

    image_paths = []
    for i in range(n):
        frame_name = frame_files[i]
        stem       = os.path.splitext(frame_name)[0]
        label_path = os.path.join(labels_dir, stem + '.txt')

        # Image path via symlink — /images/ triggers YOLOv5 label resolution
        image_paths.append(os.path.join(images_link, frame_name))

        if exists[i] == 0:
            open(label_path, 'w').close()
        else:
            x, y, w, h = gt_rects[i]
            cx_norm = (x + w / 2.0) / img_w
            cy_norm = (y + h / 2.0) / img_h
            w_norm  = w / img_w
            h_norm  = h / img_h
            cx_norm = max(0.0, min(1.0, cx_norm))
            cy_norm = max(0.0, min(1.0, cy_norm))
            w_norm  = max(0.0, min(1.0, w_norm))
            h_norm  = max(0.0, min(1.0, h_norm))
            with open(label_path, 'w') as lf:
                lf.write(f'0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n')

    return image_paths


def process_split(split_name: str, dataset_root: str) -> None:
    split_dir = os.path.join(dataset_root, split_name)
    if not os.path.isdir(split_dir):
        print(f'WARNING: {split_dir} does not exist, skipping {split_name}')
        return

    sequences = sorted(
        d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
    )
    print(f'\n=== {split_name}: {len(sequences)} sequences ===')

    all_image_paths = []
    for i, seq in enumerate(sequences, 1):
        seq_dir = os.path.join(split_dir, seq)
        paths   = process_sequence(seq_dir)
        all_image_paths.extend(paths)
        print(f'  [{i:03d}/{len(sequences)}] {seq}: {len(paths)} frames')

    list_file = os.path.join(dataset_root, f'{split_name}_rgb_images.txt')
    with open(list_file, 'w') as f:
        f.write('\n'.join(all_image_paths) + '\n')
    print(f'\nWrote {list_file}  ({len(all_image_paths)} images, paths via rgb_yolo/images symlink)')


def main():
    parser = argparse.ArgumentParser(
        description='Prepare YOLO labels from Anti-UAV RGB (visible) JSON annotations'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        choices=['train', 'val'],
        help='Splits to process (default: train val)',
    )
    parser.add_argument(
        '--dataset_root',
        default=DATASET_ROOT,
        help='Path to Anti-UAV-RGBT dataset root',
    )
    args = parser.parse_args()

    for split in args.splits:
        process_split(split, args.dataset_root)

    print('\nAll done.')


if __name__ == '__main__':
    main()
