#!/usr/bin/env python3
"""Validate a trained Stage-1 Gaussian checkpoint without retraining."""

import argparse
import math
from pathlib import Path

from yolov5.utils.torch_utils import select_device

from lib.models.yolo_evidential.model import EvidentialYOLO
from lib.models.yolo_evidential.train_single_stream import build_loaders, validate
from lib.models.yolo_evidential.train_stage1_gaussian import check_variance_health


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-txt", type=Path, required=True)
    parser.add_argument("--val-txt", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    device = select_device(args.device)
    model = EvidentialYOLO.load(
        str(args.checkpoint), str(args.base_weights), device=device, K=2
    )
    stride = int(model.stride.max())
    _, val_loader, _ = build_loaders(
        str(args.train_txt), str(args.val_txt), str(args.data),
        640, args.batch_size, stride, workers=args.workers,
        hyp={"mosaic": 0.0},
    )
    model.model[-1].use_cls_for_conf = True
    metrics = validate(model, val_loader, device, conf_thres=0.25, iou_thres=0.45, K=2)
    health = check_variance_health(model, val_loader, device)

    print(
        f"mAP50={metrics['mAP50']:.6f} "
        f"mAP50-95={metrics['mAP5095']:.6f} "
        f"variance_mean={health['sigma2_mean']:.6f} "
        f"variance_std={health['sigma2_std']:.6f}"
    )
    values = [metrics["mAP50"], metrics["mAP5095"], health["sigma2_mean"], health["sigma2_std"]]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite Gaussian validation result")
    if metrics["mAP50"] <= 0.0:
        raise RuntimeError("Gaussian checkpoint produced zero validation mAP50")
    if not health["healthy"]:
        raise RuntimeError("Gaussian checkpoint failed variance-health checks")
    print("GAUSSIAN CHECKPOINT VERIFICATION PASSED")


if __name__ == "__main__":
    main()
