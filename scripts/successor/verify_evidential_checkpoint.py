#!/usr/bin/env python3
"""Validate and numerically verify a selected Stage-2 checkpoint."""

import argparse
import math
from pathlib import Path

from yolov5.utils.torch_utils import select_device

from lib.models.yolo_evidential.model import EvidentialYOLO
from lib.models.yolo_evidential.train_single_stream import build_loaders, validate
from lib.models.yolo_evidential.verify import run_verification


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

    # Stage 2 and downstream fusion use objectness times learned UAV belief.
    model.model[-1].use_cls_for_conf = False
    metrics = validate(model, val_loader, device, conf_thres=0.25, iou_thres=0.45, K=2)
    for images, *_ in val_loader:
        report = run_verification(model, images[:8].float() / 255.0, device, K=2)
        break

    print(
        f"mAP50={metrics['mAP50']:.6f} "
        f"mAP50-95={metrics['mAP5095']:.6f} "
        f"u_mean={metrics['u_mean']:.6f} "
        f"b_uav_mean={metrics['b_uav_mean']:.6f}"
    )
    values = [
        metrics["mAP50"], metrics["mAP5095"], metrics["u_mean"],
        metrics["b_uav_mean"], report["sigma2_min"], report["sigma2_norm_min"],
    ]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite evidential validation result")
    if metrics["mAP50"] <= 0.0:
        raise RuntimeError("selected checkpoint produced zero validation mAP50")
    if any(message.startswith("[FAIL]") for message in report["warnings"]):
        raise RuntimeError("selected checkpoint failed numerical verification")
    print("EVIDENTIAL CHECKPOINT VERIFICATION PASSED")


if __name__ == "__main__":
    main()
