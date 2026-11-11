#!/usr/bin/env python3
"""Extract exact selected-anchor candidates for one validation modality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import torch

from lib.models.yolo_evidential.model import EvidentialYOLO
from lib.models.yolo_evidential.postprocess import select_successor_batch
from src.thesis_experiments.detector_contract import load_detector_pair
from src.thesis_experiments.candidate_validation import validate_inference_amendment
from src.thesis_experiments.detector_preprocessing import image_batches, PREPROCESSING_ID
from src.thesis_experiments.io import atomic_write_json
from src.thesis_experiments.provenance import sha256_file, _git
from yolov5.utils.general import check_img_size, scale_boxes
from yolov5.utils.torch_utils import select_device


EXPECTED_VALIDATION_FRAMES = 61999
LOG_VARIANCE_MIN = -10.0
LOG_VARIANCE_MAX = 10.0


def _expected_images(manifest: Path, derived_root: Path, modality: str) -> dict[Path, str]:
    result: dict[Path, str] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("split") != "val":
                continue
            sensor = row.get(modality)
            if not isinstance(sensor, dict) or not isinstance(sensor.get("image"), str):
                raise ValueError(f"missing {modality} image at manifest line {line_number}")
            image = (derived_root / sensor["image"]).resolve()
            if image in result:
                raise ValueError(f"duplicate validation image in manifest: {image}")
            result[image] = row["key"]
    if len(result) != EXPECTED_VALIDATION_FRAMES:
        raise ValueError(
            f"expected {EXPECTED_VALIDATION_FRAMES} validation images, found {len(result)}"
        )
    return result


def _values(tensor: torch.Tensor) -> list[float]:
    values = [float(value) for value in tensor.detach().cpu().tolist()]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("attempted to serialize non-finite tensor")
    return values


def _candidate_record(selected: dict, image, shape) -> tuple[dict, int, int]:
    native_xyxy = scale_boxes(
        image.shape[1:],
        selected["box_xyxy_input"].detach().clone().reshape(1, 4),
        shape[0],
        shape[1],
    )[0]
    x1, y1, x2, y2 = _values(native_xyxy)
    native_xywh = [x1, y1, x2 - x1, y2 - y1]
    if native_xywh[2] <= 0.0 or native_xywh[3] <= 0.0:
        raise RuntimeError("selected candidate became empty in native coordinates")

    grid_variance = _values(selected["sigma2_grid"])
    lower = math.exp(LOG_VARIANCE_MIN)
    upper = math.exp(LOG_VARIANCE_MAX)
    lower_saturated = sum(value <= lower * (1.0 + 1e-6) for value in grid_variance)
    upper_saturated = sum(value >= upper * (1.0 - 1e-6) for value in grid_variance)
    corrected_components = _values(selected["sigma2_norm_corrected"])
    record = {
        "anchor_index": int(selected["anchor_index"]),
        "candidate_score": float(selected["candidate_score"].item()),
        "objectness": float(selected["objectness"].item()),
        "beliefs": _values(selected["beliefs"]),
        "semantic_uncertainty": float(selected["uncertainty"].item()),
        "box_xywh_input": _values(selected["box_xywh_input"]),
        "box_xywh_native": native_xywh,
        "sigma2_grid": grid_variance,
        "sigma2_pixel": _values(selected["sigma2_pixel"]),
        "sigma2_norm_corrected_components": corrected_components,
        "spatial_uncertainty_corrected": sum(corrected_components) / 4.0,
        "sigma2_norm_inherited_head_components": _values(
            selected["sigma2_norm_inherited_head"]
        ),
        "log_variance_clip_hits": {
            "lower": lower_saturated,
            "upper": upper_saturated,
        },
    }
    if not all(math.isfinite(value) for value in (
        record["candidate_score"], record["objectness"],
        record["semantic_uncertainty"], record["spatial_uncertainty_corrected"],
    )):
        raise RuntimeError("non-finite scalar candidate output")
    return record, lower_saturated, upper_saturated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-pair-manifest", type=Path, required=True)
    parser.add_argument("--inference-amendment", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--modality", choices=("rgb", "thermal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if args.output.exists() or args.summary.exists():
        raise FileExistsError("refusing to replace an existing extraction artifact")
    pair = load_detector_pair(args.detector_pair_manifest)
    validate_inference_amendment(args.inference_amendment)
    if args.imgsz != 640:
        raise ValueError("the inference amendment requires imgsz=640")
    expected = _expected_images(pair.development_manifest_path, args.derived_root, args.modality)
    amendment_hash = sha256_file(args.inference_amendment)
    device = select_device(args.device)
    model = EvidentialYOLO.load(
        str(pair.selected_checkpoints[args.modality]),
        str(pair.clean_checkpoints[args.modality]),
        device=device,
        K=2,
    ).eval()
    model.model[-1].use_cls_for_conf = False
    stride = int(model.stride.max())
    imgsz = check_img_size(args.imgsz, s=max(stride, 32))
    loader = image_batches(sorted(expected), batch_size=args.batch_size,
                           workers=args.workers, imgsz=imgsz, stride=stride)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=str(args.output.parent)
    )
    digest = hashlib.sha256()
    counts = Counter()
    seen: set[str] = set()
    try:
        with os.fdopen(descriptor, "wb") as output, torch.inference_mode():
            for batch_index, (images, _targets, paths, shapes) in enumerate(loader, 1):
                images = images.to(device, non_blocking=True).float() / 255.0
                model_output = model(images)
                selected_batch = select_successor_batch(
                    model_output[0], model_output[4], score_minimum=0.25
                )
                for index, (path, shape, selected) in enumerate(
                    zip(paths, shapes, selected_batch)
                ):
                    resolved = Path(path).resolve()
                    key = expected.get(resolved)
                    if key is None:
                        raise RuntimeError(f"loader image is outside frozen manifest: {resolved}")
                    if key in seen:
                        raise RuntimeError(f"duplicate extracted key: {key}")
                    seen.add(key)
                    row = {
                        "schema_version": 1,
                        "key": key,
                        "modality": args.modality,
                        "candidate": None,
                    }
                    if selected is None:
                        counts["missing"] += 1
                    else:
                        candidate, lower_hits, upper_hits = _candidate_record(
                            selected, images[index], shape
                        )
                        row["candidate"] = candidate
                        counts["selected"] += 1
                        counts["lower_clip_coordinate_hits"] += lower_hits
                        counts["upper_clip_coordinate_hits"] += upper_hits
                    content = (
                        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    ).encode()
                    output.write(content)
                    digest.update(content)
                    counts["frames"] += 1
                if batch_index % 50 == 0:
                    print(f"batches={batch_index} frames={counts['frames']}", flush=True)
            output.flush()
            os.fsync(output.fileno())
        if seen != set(expected.values()):
            missing = len(set(expected.values()) - seen)
            raise RuntimeError(f"extraction omitted {missing} frozen validation keys")
        if counts["frames"] != EXPECTED_VALIDATION_FRAMES:
            raise RuntimeError(f"unexpected extracted frame count: {counts['frames']}")
        os.replace(temporary_name, args.output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    summary = {
        "schema_version": 1,
        "modality": args.modality,
        "output": str(args.output.resolve()),
        "output_sha256": digest.hexdigest(),
        "counts": dict(sorted(counts.items())),
        "detector_pair_manifest_sha256": pair.manifest_sha256,
        "selected_checkpoint_sha256": sha256_file(pair.selected_checkpoints[args.modality]),
        "clean_checkpoint_sha256": sha256_file(pair.clean_checkpoints[args.modality]),
        "development_manifest_sha256": sha256_file(pair.development_manifest_path),
        "inference_amendment_sha256": amendment_hash,
        "score_minimum": 0.25,
        "score_minimum_inclusive": True,
        "inference_log_variance_clip": [LOG_VARIANCE_MIN, LOG_VARIANCE_MAX],
        "preprocessing": PREPROCESSING_ID,
        "imgsz": imgsz,
        "stride": stride,
        "batch_size": args.batch_size,
        "source_provenance": source_provenance(),
    }
    atomic_write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"VALIDATION CANDIDATE EXTRACTION PASSED: {args.modality}")


def source_provenance():
    repository = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), repository / "src/thesis_experiments/detector_preprocessing.py",
               repository / "resources/EDTC/lib/models/yolo_evidential/postprocess.py",
               repository / "resources/EDTC/lib/models/yolo_evidential/detect_head.py",
               repository / "resources/EDTC/lib/models/yolo_evidential/model.py",
               repository / "resources/EDTC/lib/utils/spatial_uncertainty.py",
               repository / "resources/EDTC/yolov5/utils/augmentations.py",
               repository / "resources/EDTC/yolov5/utils/general.py"]
    return {
        "git_commit": _git(repository, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(repository, "status", "--porcelain")),
        "source_sha256": {str(path.relative_to(repository)): sha256_file(path)
                          for path in sources},
        "python": __import__("sys").version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


if __name__ == "__main__":
    main()
