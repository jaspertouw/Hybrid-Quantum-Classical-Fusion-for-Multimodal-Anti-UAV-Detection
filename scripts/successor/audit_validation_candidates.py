#!/usr/bin/env python3
"""Merge validation candidate shards and compare corrected versus legacy DMS."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.thesis_experiments.detector_contract import load_detector_pair
from src.thesis_experiments.io import atomic_write_json
from src.thesis_experiments.paired_evaluation import (
    Candidate,
    aggregate_frame_weighted,
    corrected_dms_winner,
    evaluate_legacy_dms_sequence,
    evaluate_sequence,
    legacy_dms_score,
    legacy_dms_winner,
)
from src.thesis_experiments.provenance import sha256_file


EXPECTED_VALIDATION_FRAMES = 61999


def _load_shard(path: Path, modality: str) -> dict[str, dict | None]:
    rows: dict[str, dict | None] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("schema_version") != 1 or row.get("modality") != modality:
                raise ValueError(f"invalid {modality} shard row {line_number}")
            key = row.get("key")
            candidate = row.get("candidate")
            if not isinstance(key, str) or key in rows:
                raise ValueError(f"invalid or duplicate key in {modality} shard: {key}")
            if candidate is not None:
                if not isinstance(candidate, dict):
                    raise ValueError(f"invalid candidate at {modality}/{key}")
                beliefs = candidate.get("beliefs")
                uncertainty = candidate.get("semantic_uncertainty")
                if not isinstance(beliefs, list) or len(beliefs) != 2:
                    raise ValueError(f"invalid beliefs at {modality}/{key}")
                values = [*beliefs, uncertainty, candidate.get("objectness"),
                          candidate.get("candidate_score"),
                          candidate.get("spatial_uncertainty_corrected")]
                if not all(isinstance(value, (int, float)) and math.isfinite(value)
                           for value in values):
                    raise ValueError(f"non-finite candidate at {modality}/{key}")
                if abs(sum(beliefs) + uncertainty - 1.0) > 1e-5:
                    raise ValueError(f"belief partition mismatch at {modality}/{key}")
            rows[key] = candidate
    if len(rows) != EXPECTED_VALIDATION_FRAMES:
        raise ValueError(f"expected {EXPECTED_VALIDATION_FRAMES} {modality} rows, found {len(rows)}")
    return rows


def _load_validation_frames(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "val":
                rows.append(row)
    if len(rows) != EXPECTED_VALIDATION_FRAMES:
        raise ValueError(f"expected {EXPECTED_VALIDATION_FRAMES} manifest rows, found {len(rows)}")
    return rows


def _candidate(sensor: str, value: dict | None) -> Candidate | None:
    if value is None:
        return None
    box = value.get("box_xywh_native")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"invalid native box for {sensor}")
    return Candidate(sensor, tuple(float(item) for item in box))


def _mean(values) -> float:
    values = tuple(values)
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-pair-manifest", type=Path, required=True)
    parser.add_argument("--rgb-shard", type=Path, required=True)
    parser.add_argument("--thermal-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace audit artifact: {args.output}")

    pair = load_detector_pair(args.detector_pair_manifest)
    frames = _load_validation_frames(pair.development_manifest_path)
    rgb = _load_shard(args.rgb_shard, "rgb")
    thermal = _load_shard(args.thermal_shard, "thermal")
    expected_keys = {frame["key"] for frame in frames}
    if set(rgb) != expected_keys or set(thermal) != expected_keys:
        raise RuntimeError("candidate shard membership differs from frozen validation manifest")

    by_sequence: dict[str, list[tuple[dict, Candidate | None, Candidate | None]]] = defaultdict(list)
    counts = Counter()
    for frame in frames:
        key = frame["key"]
        rgb_value, thermal_value = rgb[key], thermal[key]
        rgb_candidate = _candidate("rgb", rgb_value)
        thermal_candidate = _candidate("thermal", thermal_value)
        corrected_sensor = corrected_dms_winner(
            None if rgb_value is None else float(rgb_value["spatial_uncertainty_corrected"]),
            None if thermal_value is None else float(thermal_value["spatial_uncertainty_corrected"]),
        )
        rgb_legacy_score = None if rgb_value is None else legacy_dms_score(
            rgb_value["sigma2_grid"], rgb_value["box_xywh_native"]
        )
        thermal_legacy_score = None if thermal_value is None else legacy_dms_score(
            thermal_value["sigma2_grid"], thermal_value["box_xywh_native"]
        )
        legacy_sensor = legacy_dms_winner(rgb_legacy_score, thermal_legacy_score)
        candidates = {"rgb": rgb_candidate, "thermal": thermal_candidate, None: None}
        corrected_candidate = candidates[corrected_sensor]
        legacy_candidate = candidates[legacy_sensor]
        by_sequence[frame["sequence"]].append(
            (frame, corrected_candidate, legacy_candidate)
        )
        counts["frames"] += 1
        counts[f"availability_{'both' if rgb_candidate and thermal_candidate else 'rgb_only' if rgb_candidate else 'thermal_only' if thermal_candidate else 'neither'}"] += 1
        counts[f"corrected_winner_{corrected_sensor or 'none'}"] += 1
        counts[f"legacy_winner_{legacy_sensor or 'none'}"] += 1
        if rgb_candidate is not None and thermal_candidate is not None:
            counts["both_available"] += 1
            counts["winner_disagreements"] += int(corrected_sensor != legacy_sensor)

    corrected_results = []
    legacy_winner_methodology_results = []
    released_legacy_results = []
    sequence_rows = []
    for sequence in sorted(by_sequence):
        entries = by_sequence[sequence]
        sequence_frames = [entry[0] for entry in entries]
        corrected_candidates = [entry[1] for entry in entries]
        legacy_candidates = [entry[2] for entry in entries]
        corrected_result = evaluate_sequence(sequence_frames, corrected_candidates)
        legacy_methodology_result = evaluate_sequence(sequence_frames, legacy_candidates)
        released_result = evaluate_legacy_dms_sequence(sequence_frames, legacy_candidates)
        corrected_results.append(corrected_result)
        legacy_winner_methodology_results.append(legacy_methodology_result)
        released_legacy_results.append(released_result)
        sequence_rows.append({
            "sequence": sequence,
            "frames": corrected_result.frames,
            "paired_present": corrected_result.present,
            "corrected_methodology_acc": corrected_result.accuracy,
            "legacy_winner_methodology_acc": legacy_methodology_result.accuracy,
            "released_legacy_acc": released_result.accuracy,
        })
    if len(sequence_rows) != 67:
        raise RuntimeError(f"expected 67 validation sequences, found {len(sequence_rows)}")

    corrected_pooled = aggregate_frame_weighted(corrected_results)
    legacy_methodology_pooled = aggregate_frame_weighted(legacy_winner_methodology_results)
    result = {
        "schema_version": 1,
        "status": "validation_audit_not_test_result",
        "detector_pair_manifest_sha256": pair.manifest_sha256,
        "development_manifest_sha256": sha256_file(pair.development_manifest_path),
        "candidate_shards": {
            "rgb": {"path": str(args.rgb_shard.resolve()), "sha256": sha256_file(args.rgb_shard)},
            "thermal": {"path": str(args.thermal_shard.resolve()), "sha256": sha256_file(args.thermal_shard)},
        },
        "counts": dict(sorted(counts.items())),
        "winner_disagreement_rate_when_both_available": (
            counts["winner_disagreements"] / counts["both_available"]
            if counts["both_available"] else None
        ),
        "sequence_mean": {
            "corrected_winner_methodology_edtc": _mean(r.accuracy for r in corrected_results),
            "legacy_winner_methodology_edtc": _mean(
                r.accuracy for r in legacy_winner_methodology_results
            ),
            "released_legacy_winner_and_evaluator_edtc": _mean(
                r.accuracy for r in released_legacy_results
            ),
        },
        "frame_weighted": {
            "corrected_winner_methodology_edtc": corrected_pooled.accuracy,
            "legacy_winner_methodology_edtc": legacy_methodology_pooled.accuracy,
        },
        "sequences": sequence_rows,
    }
    atomic_write_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "counts", "winner_disagreement_rate_when_both_available",
        "sequence_mean", "frame_weighted"
    )}, indent=2, sort_keys=True))
    print(f"audit_sha256={sha256_file(args.output)}")
    print("PAIRED VALIDATION AUDIT PASSED")


if __name__ == "__main__":
    main()
