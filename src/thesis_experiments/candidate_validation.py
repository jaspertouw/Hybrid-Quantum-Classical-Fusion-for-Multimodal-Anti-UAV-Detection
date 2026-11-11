"""Strict validation of candidate values and extraction artifact provenance."""

import json
import math

from .provenance import sha256_file
from .detector_contract import EXPECTED_PROTOCOL_SHA256


def validate_inference_amendment(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["amendment"]["base_protocol"]["sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("inference amendment references a different base protocol")
    if document["preprocessing"]["id"] != "per-image-yolo-letterbox-auto-v1":
        raise ValueError("unsupported inference preprocessing contract")
    if document["preprocessing"]["image_size"] != 640:
        raise ValueError("unsupported inference image size")
    selection = document["candidate_selection"]
    if selection["minimum"] != 0.25 or selection["minimum_is_inclusive"] is not True:
        raise ValueError("unsupported inference score boundary")
    if document["variance"]["inference_log_variance_clip"] != [-10.0, 10.0]:
        raise ValueError("unsupported inference variance bounds")
    return document


def validate_candidate(value):
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("candidate must be an object or null")

    def scalar(key, low=None, high=None):
        number = value.get(key)
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number)):
            raise ValueError(f"invalid candidate scalar: {key}")
        if (low is not None and number < low) or (high is not None and number > high):
            raise ValueError(f"candidate scalar outside range: {key}")
        return number

    def vector(key, length, minimum=None):
        numbers = value.get(key)
        if not isinstance(numbers, list) or len(numbers) != length:
            raise ValueError(f"invalid candidate vector: {key}")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) and (minimum is None or v >= minimum)
                   for v in numbers):
            raise ValueError(f"invalid candidate vector values: {key}")
        return numbers

    beliefs = vector("beliefs", 2, 0)
    uncertainty = scalar("semantic_uncertainty", 0, 1)
    if uncertainty <= 0 or abs(sum(beliefs) + uncertainty - 1) > 1e-5:
        raise ValueError("candidate belief partition mismatch")
    obj = scalar("objectness", 0, 1)
    score = scalar("candidate_score", 0.25, 1)
    if not math.isclose(score, obj * beliefs[0], rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError("candidate confidence mismatch")
    for key in ("box_xywh_native", "box_xywh_input"):
        box = vector(key, 4)
        if box[2] <= 0 or box[3] <= 0:
            raise ValueError(f"non-positive candidate dimensions: {key}")
    grid = vector("sigma2_grid", 4, 0)
    pixel = vector("sigma2_pixel", 4, 0)
    corrected = vector("sigma2_norm_corrected_components", 4, 0)
    vector("sigma2_norm_inherited_head_components", 4, 0)
    if min(grid) <= 0 or min(pixel) <= 0:
        raise ValueError("non-positive candidate variance")
    ratios = [p / g for p, g in zip(pixel, grid)]
    if not all(math.isclose(r, ratios[0], rel_tol=1e-5) for r in ratios):
        raise ValueError("inconsistent variance unit conversion")
    width, height = value["box_xywh_input"][2:]
    denominators = (max(width, 1)**2, max(height, 1)**2) * 2
    if not all(math.isclose(c, p / d, rel_tol=2e-5, abs_tol=1e-8)
               for c, p, d in zip(corrected, pixel, denominators)):
        raise ValueError("corrected variance normalization mismatch")
    spatial = scalar("spatial_uncertainty_corrected", 0)
    if not math.isclose(spatial, sum(corrected) / 4, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError("spatial scalar mismatch")
    anchor = value.get("anchor_index")
    if type(anchor) is not int or anchor < 0:
        raise ValueError("invalid anchor index")


def verify_shard_summary(path, modality, pair):
    summary_path = path.with_name(f"{modality}_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "modality": modality,
        "output_sha256": sha256_file(path),
        "detector_pair_manifest_sha256": pair.manifest_sha256,
        "development_manifest_sha256": pair.document["development_manifest"]["sha256"],
        "selected_checkpoint_sha256": pair.document["modalities"][modality]["selected"]["sha256"],
        "clean_checkpoint_sha256": pair.document["modalities"][modality]["clean"]["sha256"],
        "score_minimum": 0.25,
        "score_minimum_inclusive": True,
        "inference_log_variance_clip": [-10.0, 10.0],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"{modality} summary mismatch: {key}")
    return summary, {"path": str(summary_path.resolve()), "sha256": sha256_file(summary_path)}
