#!/usr/bin/env python3
"""Merge validation candidate shards and compare corrected versus legacy DMS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.thesis_experiments.detector_contract import load_detector_pair
from src.thesis_experiments.candidate_validation import validate_candidate, verify_shard_summary
from src.thesis_experiments.io import atomic_write_json
from src.thesis_experiments.paired_evaluation import (
    Candidate,
    aggregate_frame_weighted,
    aggregate_pooled_components,
    corrected_dms_winner,
    evaluate_legacy_dms_sequence,
    evaluate_sequence,
    legacy_dms_score,
    legacy_dms_winner,
    legacy_native_box,
)
from src.thesis_experiments.provenance import sha256_file, _git


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
            validate_candidate(candidate)
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


def _candidate(sensor: str, value: dict | None, *, legacy=False) -> Candidate | None:
    if value is None:
        return None
    box = value.get("box_xywh_native")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"invalid native box for {sensor}")
    box = tuple(float(item) for item in box)
    return Candidate(sensor, legacy_native_box(box) if legacy else box)


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
    summaries = {}
    summary_records = {}
    for modality, path in (("rgb", args.rgb_shard), ("thermal", args.thermal_shard)):
        summaries[modality], summary_records[modality] = verify_shard_summary(path, modality, pair)
    for key in ("inference_amendment_sha256", "preprocessing", "precision", "batch_size", "imgsz", "stride"):
        if summaries["rgb"].get(key) != summaries["thermal"].get(key):
            raise RuntimeError(f"modality extraction contracts differ: {key}")
    source_rgb = summaries["rgb"].get("source_provenance")
    source_thermal = summaries["thermal"].get("source_provenance")
    if bool(source_rgb) != bool(source_thermal):
        raise RuntimeError("only one modality has source provenance")
    if source_rgb:
        for key in ("git_commit", "source_sha256", "pytorch", "cuda_runtime"):
            if source_rgb.get(key) != source_thermal.get(key):
                raise RuntimeError(f"modality source provenance differs: {key}")
        if source_rgb.get("git_dirty") or source_thermal.get("git_dirty"):
            raise RuntimeError("extraction was performed from a dirty repository")
    frames = _load_validation_frames(pair.development_manifest_path)
    protocol = json.loads(pair.protocol_path.read_text(encoding="utf-8"))
    membership = sorted({frame["sequence"] for frame in frames})
    membership_hash = hashlib.sha256("".join(name + "\n" for name in membership).encode()).hexdigest()
    if membership_hash != protocol["dataset"]["membership_sha256"]["val"]:
        raise RuntimeError("validation sequence membership hash differs from protocol")
    for frame in frames:
        if frame["paired_presence"] != (frame["rgb"]["present"] or frame["thermal"]["present"]):
            raise RuntimeError(f"paired-presence mismatch: {frame['key']}")
    rgb = _load_shard(args.rgb_shard, "rgb")
    thermal = _load_shard(args.thermal_shard, "thermal")
    expected_keys = {frame["key"] for frame in frames}
    if len(expected_keys) != len(frames):
        raise RuntimeError("duplicate frozen validation key")
    if set(rgb) != expected_keys or set(thermal) != expected_keys:
        raise RuntimeError("candidate shard membership differs from frozen validation manifest")
    for modality, shard in (("rgb", rgb), ("thermal", thermal)):
        selected_count = sum(value is not None for value in shard.values())
        expected_counts = {"frames": len(shard), "selected": selected_count,
                           "missing": len(shard) - selected_count}
        for key, value in expected_counts.items():
            if summaries[modality]["counts"].get(key) != value:
                raise RuntimeError(f"{modality} summary count mismatch: {key}")

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
        # Original runner uses strict > 0.25 and rounded endpoints plus one.
        legacy_rgb_value = rgb_value if rgb_value and rgb_value["candidate_score"] > 0.25 else None
        legacy_thermal_value = thermal_value if thermal_value and thermal_value["candidate_score"] > 0.25 else None
        legacy_rgb = _candidate("rgb", legacy_rgb_value, legacy=True)
        legacy_thermal = _candidate("thermal", legacy_thermal_value, legacy=True)
        rgb_legacy_score = None if legacy_rgb is None else legacy_dms_score(
            rgb_value["sigma2_grid"], legacy_rgb.box_xywh
        )
        thermal_legacy_score = None if legacy_thermal is None else legacy_dms_score(
            thermal_value["sigma2_grid"], legacy_thermal.box_xywh
        )
        legacy_sensor = legacy_dms_winner(rgb_legacy_score, thermal_legacy_score)
        candidates = {"rgb": rgb_candidate, "thermal": thermal_candidate, None: None}
        corrected_candidate = candidates[corrected_sensor]
        legacy_candidate = {"rgb": legacy_rgb, "thermal": legacy_thermal, None: None}[legacy_sensor]
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
            "legacy_conventions_acc": released_result.accuracy,
        })
    if len(sequence_rows) != 67:
        raise RuntimeError(f"expected 67 validation sequences, found {len(sequence_rows)}")

    corrected_pooled = aggregate_pooled_components(corrected_results)
    legacy_methodology_pooled = aggregate_pooled_components(legacy_winner_methodology_results)
    result = {
        "schema_version": 2,
        "audit_source": {
            "git_commit": _git(Path(__file__).resolve().parents[2], "rev-parse", "HEAD"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "evaluator_sha256": sha256_file(Path(__file__).resolve().parents[2] / "src/thesis_experiments/paired_evaluation.py"),
            "validator_sha256": sha256_file(Path(__file__).resolve().parents[2] / "src/thesis_experiments/candidate_validation.py"),
        },
        "status": "validation_audit_not_test_result",
        "legacy_reference_scope": "released rounding, strict threshold, DMS and evaluator on extracted candidates; original preprocessing/NMS tie parity requires separate verification",
        "extraction_summaries": summary_records,
        "source_provenance_available": bool(source_rgb),
        "preprocessing": summaries["rgb"].get("preprocessing", "historical-rectangular-loader"),
        "inference_amendment_sha256": summaries["rgb"]["inference_amendment_sha256"],
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
            "legacy_geometry_winner_and_evaluator_edtc": _mean(
                r.accuracy for r in released_legacy_results
            ),
        },
        "frame_weighted": {
            "corrected_winner_methodology_edtc": aggregate_frame_weighted(corrected_results),
            "legacy_winner_methodology_edtc": aggregate_frame_weighted(legacy_winner_methodology_results),
        },
        "pooled_components_diagnostic": {
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
