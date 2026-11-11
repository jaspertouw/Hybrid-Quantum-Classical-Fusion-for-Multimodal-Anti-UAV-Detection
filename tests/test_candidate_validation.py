import copy
import json
from types import SimpleNamespace

import pytest

from src.thesis_experiments.candidate_validation import (
    validate_candidate, validate_inference_amendment, verify_shard_summary,
)
from src.thesis_experiments.provenance import sha256_file
from pathlib import Path


def valid_candidate():
    return {
        "beliefs": [0.6, 0.1], "semantic_uncertainty": 0.3,
        "objectness": 0.8, "candidate_score": 0.48, "anchor_index": 1,
        "box_xywh_input": [4, 4, 8, 4], "box_xywh_native": [0, 0, 8, 4],
        "sigma2_grid": [1, 1, 1, 1], "sigma2_pixel": [64, 64, 64, 64],
        "sigma2_norm_corrected_components": [1, 4, 1, 4],
        "sigma2_norm_inherited_head_components": [0.125, 0.25, 0.015625, 0.0625],
        "spatial_uncertainty_corrected": 2.5,
    }


def test_correct_candidate_and_missing_are_accepted():
    validate_candidate(valid_candidate())
    validate_candidate(None)


@pytest.mark.parametrize("key,value,match", [
    ("candidate_score", 0.5, "confidence mismatch"),
    ("objectness", 1.2, "outside range"),
    ("beliefs", [0.6, 0.4], "partition mismatch"),
    ("sigma2_grid", [0, 1, 1, 1], "non-positive"),
    ("sigma2_pixel", [32, 64, 64, 64], "unit conversion"),
    ("sigma2_norm_corrected_components", [1, 1, 1, 1], "normalization mismatch"),
    ("spatial_uncertainty_corrected", float("nan"), "invalid candidate scalar"),
    ("box_xywh_native", [0, 0, 0, 1], "non-positive"),
])
def test_corrupt_candidates_fail_before_scoring(key, value, match):
    candidate = copy.deepcopy(valid_candidate())
    candidate[key] = value
    with pytest.raises(ValueError, match=match):
        validate_candidate(candidate)


def test_summary_file_hash_is_verified(tmp_path):
    shard = tmp_path / "rgb.jsonl"
    shard.write_text("{}\n")
    document = {"development_manifest": {"sha256": "dev"}, "modalities": {
        "rgb": {"selected": {"sha256": "selected"}, "clean": {"sha256": "clean"}}}}
    pair = SimpleNamespace(document=document, manifest_sha256="pair")
    summary = {
        "schema_version": 1, "modality": "rgb", "output_sha256": sha256_file(shard),
        "detector_pair_manifest_sha256": "pair", "development_manifest_sha256": "dev",
        "selected_checkpoint_sha256": "selected", "clean_checkpoint_sha256": "clean",
        "score_minimum": 0.25, "score_minimum_inclusive": True,
        "inference_log_variance_clip": [-10.0, 10.0],
    }
    (tmp_path / "rgb_summary.json").write_text(json.dumps(summary))
    verify_shard_summary(shard, "rgb", pair)
    shard.write_text("altered content\n")
    with pytest.raises(ValueError, match="output_sha256"):
        verify_shard_summary(shard, "rgb", pair)


def test_active_amendment_matches_canonical_preprocessing():
    document = validate_inference_amendment(Path("experiments/protocol/inference_v1.json"))
    assert document["amendment"]["version"] == "1.1.0"
