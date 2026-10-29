"""
Convert Anti-UAV IR JSON annotations to YOLO label format for YOLOv5 training.

For each sequence in the requested splits (train / val), this script:
  1. Writes per-frame YOLO .txt label files into <seq_dir>/labels/
  2. Creates a symlink <seq_dir>/images -> ir_frames  (YOLOv5 label resolution
     requires '/images/' in the image path so it can substitute '/labels/')
  3. Writes image-list .txt files at the dataset root using 'images/' paths

YOLOv5 label resolution (dataloaders.py):
    path.replace('/images/', '/labels/').replace('.jpg', '.txt')
  → .../images/000001.jpg  →  .../labels/000001.txt  ✓

Usage (run on Snellius, see prepare_yolo_labels.job):
    python scripts/prepare_yolo_labels.py
    python scripts/prepare_yolo_labels.py --splits train
    python scripts/prepare_yolo_labels.py --splits train val
"""

import argparse
import json
import os

import cv2


DATASET_ROOT = '/home/ssuttorp/Anti-UAV-RGBT'


def get_image_size(ir_frames_dir: str):
    """Return (width, height) by reading the first frame in ir_frames_dir."""
    for fname in sorted(os.listdir(ir_frames_dir)):
        if fname.lower().endswith('.jpg'):
            img = cv2.imread(os.path.join(ir_frames_dir, fname))
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    raise RuntimeError(f'No readable JPEG found in {ir_frames_dir}')


def process_sequence(seq_dir: str) -> list:
    """
    Write YOLO label files and create images/ symlink for one sequence.

    Returns a sorted list of absolute image paths using the images/ symlink,
    so YOLOv5's label resolver can substitute /images/ -> /labels/.
    """
    ir_frames_dir = os.path.join(seq_dir, 'ir_frames')
    images_link   = os.path.join(seq_dir, 'images')   # symlink -> ir_frames
    label_json    = os.path.join(seq_dir, 'IR_label.json')
    labels_dir    = os.path.join(seq_dir, 'labels')

    if not os.path.isdir(ir_frames_dir):
        print(f'  WARNING: ir_frames/ missing in {seq_dir}, skipping')
        return []
    if not os.path.isfile(label_json):
        print(f'  WARNING: IR_label.json missing in {seq_dir}, skipping')
        return []

    # --- 1. Create images/ symlink pointing to ir_frames/ -------------------
    # YOLOv5 needs '/images/' in the path to resolve labels correctly.
    if not os.path.exists(images_link):
        os.symlink('ir_frames', images_link)

    with open(label_json, 'r') as f:
        label = json.load(f)

    gt_rects = label['gt_rect']   # list of [x, y, w, h] in pixels (top-left origin)
    exists   = label['exist']     # list of 0/1

    img_w, img_h = get_image_size(ir_frames_dir)

    # --- 2. Write YOLO label files into labels/ ------------------------------
    os.makedirs(labels_dir, exist_ok=True)

    frame_files = sorted(
        f for f in os.listdir(ir_frames_dir) if f.lower().endswith('.jpg')
    )
    n = min(len(frame_files), len(gt_rects), len(exists))

    image_paths = []
    for i in range(n):
        frame_name  = frame_files[i]                   # e.g. 000001.jpg
        stem        = os.path.splitext(frame_name)[0]  # e.g. 000001
        label_path  = os.path.join(labels_dir, stem + '.txt')

        # Image path via symlink — /images/ triggers YOLOv5 label resolution
        image_paths.append(os.path.join(images_link, frame_name))

        if exists[i] == 0:
            # No target present — write empty label file (background frame)
            open(label_path, 'w').close()
        else:
            x, y, w, h = gt_rects[i]
            # Convert [x_tl, y_tl, w, h] → YOLO [cx_norm, cy_norm, w_norm, h_norm]
            cx_norm = (x + w / 2.0) / img_w
            cy_norm = (y + h / 2.0) / img_h
            w_norm  = w / img_w
            h_norm  = h / img_h
            # Clamp to [0, 1] to guard against annotation noise
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

    list_file = os.path.join(dataset_root, f'{split_name}_ir_images.txt')
    with open(list_file, 'w') as f:
        f.write('\n'.join(all_image_paths) + '\n')
    print(f'\nWrote {list_file}  ({len(all_image_paths)} images, paths via images/ symlink)')


def main():
    parser = argparse.ArgumentParser(
        description='Prepare YOLO labels from Anti-UAV IR JSON annotations'
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
