#!/usr/bin/env python3
"""Validation-only parity checks for canonical preprocessing and legacy NMS."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from lib.models.yolo_evidential.model import EvidentialYOLO
from lib.models.yolo_evidential.postprocess import select_successor_batch
from scripts.successor.extract_validation_candidates import _expected_images, source_provenance
from src.thesis_experiments.detector_contract import load_detector_pair
from src.thesis_experiments.detector_preprocessing import prepare_image, PREPROCESSING_ID
from src.thesis_experiments.io import atomic_write_json
from src.thesis_experiments.provenance import sha256_file
from yolov5.utils.augmentations import letterbox
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-pair-manifest", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--modality", choices=("rgb", "thermal"), required=True)
    parser.add_argument("--val-txt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pair = load_detector_pair(args.detector_pair_manifest)
    expected = _expected_images(pair.development_manifest_path, args.derived_root, args.modality)
    device = select_device("0")
    model = EvidentialYOLO.load(str(pair.selected_checkpoints[args.modality]),
                                str(pair.clean_checkpoints[args.modality]), device=device, K=2).eval()
    model.model[-1].use_cls_for_conf = False
    stride = int(model.stride.max())
    _, dataset = create_dataloader(str(args.val_txt), imgsz=640, batch_size=128,
                                   stride=max(stride, 32), single_cls=True, augment=False,
                                   cache=False, rect=True, rank=-1, workers=1, shuffle=False)
    lookup = {Path(path).resolve(): i for i, path in enumerate(dataset.im_files)}
    if set(lookup) != set(expected):
        raise RuntimeError("validation loader membership differs from frozen manifest")
    sequences = defaultdict(list)
    for path, key in expected.items():
        sequences[key.split("/")[1]].append(path)
    counts = Counter()
    mismatched_keys = []
    max_batch_box_error = 0.0
    with torch.inference_mode():
        for sequence in sorted(sequences):
            paths = sorted(sequences[sequence])
            # Fixed first/middle/last sample from every sequence, before predictions.
            for index in sorted({0, len(paths) // 2, len(paths) - 1}):
                path = paths[index]
                canonical, _, _ = prepare_image(path, 640, stride)
                native = cv2.imread(str(path))
                inherited = letterbox(native, 640, stride=stride, auto=True)[0]
                inherited = np.ascontiguousarray(inherited.transpose(2, 0, 1)[::-1])
                if not np.array_equal(canonical, inherited):
                    raise RuntimeError("canonical input differs from released frame preprocessing")
                old_image = dataset[lookup[path]][0].numpy()
                counts["sampled_frames"] += 1
                if canonical.shape != old_image.shape or not np.array_equal(canonical, old_image):
                    counts["historical_loader_pixel_mismatches"] += 1
                    mismatched_keys.append(expected[path])
                tensor = torch.from_numpy(canonical).to(device).float().unsqueeze(0) / 255.0
                single = model(tensor)
                batched = model(tensor.repeat(2, 1, 1, 1))
                a = select_successor_batch(single[0], single[4])[0]
                b = select_successor_batch(batched[0], batched[4])
                for item in b:
                    if (a is None) != (item is None):
                        raise RuntimeError("candidate availability depends on inference batch size")
                    if a is not None:
                        if a["anchor_index"] != item["anchor_index"]:
                            raise RuntimeError("selected anchor depends on inference batch size")
                        for field in ("box_xywh_input", "beliefs", "uncertainty", "objectness",
                                      "sigma2_norm_corrected"):
                            if not torch.allclose(a[field], item[field], rtol=1e-4, atol=1e-5):
                                raise RuntimeError(f"batch parity tolerance exceeded: {field}")
                        error = (a["box_xywh_input"] - item["box_xywh_input"]).abs().max().item()
                        max_batch_box_error = max(max_batch_box_error, error)
                # Verify the highest strictly eligible anchor against real bundled NMS.
                detections = non_max_suppression(single[0].clone(), 0.25, 0.45,
                                                 classes=[0], max_det=300)[0]
                scores = single[0][0, :, 4] * single[0][0, :, 5]
                peak = scores.max().item()
                if peak <= 0.25:
                    if len(detections):
                        raise RuntimeError("legacy NMS accepted an ineligible score")
                    counts["legacy_missing"] += 1
                else:
                    if not len(detections):
                        raise RuntimeError("legacy NMS dropped every eligible anchor")
                    strict = select_successor_batch(single[0], single[4])[0]
                    ties = int((scores == scores.max()).sum().item())
                    counts["top_score_ties"] += int(ties > 1)
                    if ties == 1 and not torch.allclose(
                            detections[0, :4], strict["box_xyxy_input"], rtol=1e-5, atol=1e-3):
                        raise RuntimeError("direct top candidate differs from legacy NMS")
                    if abs(detections[0, 4].item() - peak) > 1e-6:
                        raise RuntimeError("legacy NMS top score mismatch")
                    counts["legacy_selected"] += 1
    result = {
        "schema_version": 1,
        "modality": args.modality,
        "preprocessing": PREPROCESSING_ID,
        "detector_pair_manifest_sha256": pair.manifest_sha256,
        "counts": dict(counts),
        "sampling": "first/middle/last frame of every validation sequence",
        "historical_loader_mismatch_keys": mismatched_keys,
        "max_batch_box_absolute_error_input_pixels": max_batch_box_error,
        "source_provenance": source_provenance(),
        "parity_script_sha256": sha256_file(Path(__file__)),
        "scope": "sampled preprocessing, batch-size and NMS top-candidate parity; not historical weights reproduction",
    }
    atomic_write_json(args.output, result)
    print(json.dumps({"modality": args.modality, "counts": result["counts"],
                      "max_batch_box_error": max_batch_box_error}, indent=2))
    print(f"parity_sha256={sha256_file(args.output)}")
    print(f"INFERENCE PARITY VERIFICATION PASSED: {args.modality}")


if __name__ == "__main__":
    main()
