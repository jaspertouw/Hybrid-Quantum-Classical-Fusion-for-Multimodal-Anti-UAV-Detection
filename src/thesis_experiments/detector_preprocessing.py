"""Canonical successor preprocessing, matching the released per-frame runner."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


PREPROCESSING_ID = "per-image-yolo-letterbox-auto-v1"


def prepare_image(path, imgsz=640, stride=32):
    # Lazy vendor import keeps cache/evaluation consumers independent of Torch.
    from yolov5.utils.augmentations import letterbox

    original = cv2.imread(str(path))
    if original is None:
        raise ValueError(f"unreadable detector image: {path}")
    image, ratio, pad = letterbox(original, imgsz, stride=stride, auto=True)
    chw = np.ascontiguousarray(image.transpose(2, 0, 1)[::-1])
    return chw, str(path), (original.shape[:2], (ratio, pad))


def image_batches(paths, batch_size=128, workers=8, imgsz=640, stride=32):
    """Batch only equal-shaped per-image inputs; padding never depends on peers."""
    import torch

    if batch_size < 1 or workers < 1:
        raise ValueError("batch size and workers must be positive")
    paths = list(paths)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(paths), batch_size):
            # Bound CPU image buffering to one batch, and keep deterministic order.
            chunk = paths[offset:offset + batch_size]
            prepared = list(executor.map(
                lambda path: prepare_image(Path(path), imgsz, stride), chunk
            ))
            groups = {}
            for image, path, shape in prepared:
                groups.setdefault(image.shape, []).append((image, path, shape))
            for group in groups.values():
                images, batch_paths, shapes = zip(*group)
                yield torch.from_numpy(np.stack(images)), None, batch_paths, shapes
