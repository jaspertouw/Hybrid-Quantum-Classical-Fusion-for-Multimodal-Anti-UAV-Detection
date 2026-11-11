"""Versioned paired caches with atomic, resumable sequence/condition shards.

This module deliberately has no Torch, OpenCV or inherited-detector imports.
Spatial features remain uncapped; missing spatial values are null until Phase 7.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from .candidate_validation import validate_candidate
from .provenance import sha256_file


SCHEMA_VERSION = 1
SENSORS = ("rgb", "thermal")
FEATURE_ORDER = tuple(f"{sensor}.{field}" for sensor in SENSORS for field in (
    "b_uav", "semantic_uncertainty", "objectness", "spatial_uncertainty"))


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def immutable_json(path, value):
    """Create complete JSON atomically; an existing value must be identical."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded(value):
            raise RuntimeError(f"immutable cache metadata differs: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded(value))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded(value):
                raise RuntimeError(f"concurrent cache metadata differs: {path}")
    finally:
        Path(temporary).unlink(missing_ok=True)


def initialize(root, context):
    context = {**context, "schema_version": SCHEMA_VERSION, "feature_order": FEATURE_ORDER,
               "feature_policy": "uncapped; missing spatial null; fit training caps in Phase 7"}
    identifier = digest(context)
    immutable_json(Path(root) / "CACHE.json", {"context_id": identifier, "context": context})
    return identifier


def features(candidate):
    return {
        "b_uav": 0.0 if candidate is None else candidate["beliefs"][0],
        "semantic_uncertainty": 1.0 if candidate is None else candidate["semantic_uncertainty"],
        "objectness": 0.0 if candidate is None else candidate["objectness"],
        "spatial_uncertainty": None if candidate is None else candidate["spatial_uncertainty_corrected"],
    }


def sensor_record(candidate, *, omitted=False, source_condition="clean", clipped_fraction=0.0,
                  noise_seed_hex=None):
    validate_candidate(candidate)
    if omitted and candidate is not None:
        raise ValueError("an omitted sensor cannot contain a candidate")
    return {"candidate": candidate, "features_uncapped": features(candidate),
            "omitted": omitted, "source_condition": source_condition,
            "clipped_fraction": clipped_fraction, "noise_seed_hex": noise_seed_hex}


def make_row(frame, condition, sensors, context_id):
    row = {"schema_version": SCHEMA_VERSION, "context_id": context_id,
           "key": frame["key"], "split": frame["split"], "sequence": frame["sequence"],
           "frame": frame["frame"], "condition": condition["id"],
           "paired_presence": frame["paired_presence"], "attributes": frame["attributes"],
           "ground_truth": {sensor: {key: frame[sensor][key]
                            for key in ("present", "box_xywh", "box_valid")} for sensor in SENSORS},
           **sensors}
    validate_row(row, condition, context_id, frame=frame)
    return row


def validate_row(row, condition, context_id, *, frame=None):
    if row.get("schema_version") != SCHEMA_VERSION or row.get("context_id") != context_id:
        raise ValueError("cache schema/provenance mismatch")
    if row.get("condition") != condition["id"]:
        raise ValueError("cache condition mismatch")
    if row.get("split") not in ("train", "val"):
        raise ValueError("development cache forbids test/unknown splits")
    if type(row.get("frame")) is not int or row["frame"] < 1:
        raise ValueError("invalid cache frame number")
    if not isinstance(row.get("attributes"), list) or not all(isinstance(tag, str) for tag in row["attributes"]):
        raise ValueError("invalid cache attributes")
    expected_key = f"{row['split']}/{row['sequence']}/{row['frame']:06d}"
    if row.get("key") != expected_key:
        raise ValueError("invalid cache frame key")
    gt = row["ground_truth"]
    if type(row.get("paired_presence")) is not bool or row["paired_presence"] != (
            gt["rgb"]["present"] or gt["thermal"]["present"]):
        raise ValueError("paired-presence label mismatch")
    for sensor in SENSORS:
        annotation = gt[sensor]
        if type(annotation["present"]) is not bool or type(annotation["box_valid"]) is not bool:
            raise ValueError("invalid sensor ground-truth flags")
        box = annotation["box_xywh"]
        if box is not None and (not isinstance(box, list) or len(box) != 4 or
                not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)):
            raise ValueError("invalid sensor ground-truth box")
        if annotation["box_valid"] != (box is not None and box[2] > 0 and box[3] > 0):
            raise ValueError("ground-truth box-valid flag mismatch")
        item = row[sensor]
        validate_candidate(item["candidate"])
        if item["features_uncapped"] != features(item["candidate"]):
            raise ValueError("cached features disagree with detector candidate")
        omitted = condition["corruption"] == "dropout" and condition["sensor"] == sensor
        if type(item["omitted"]) is not bool or item["omitted"] != omitted:
            raise ValueError("incorrect dropout encoding")
        if omitted and item["candidate"] is not None:
            raise ValueError("dropout must omit the candidate")
        source = "clean" if condition["sensor"] != sensor else condition["id"]
        if item["source_condition"] != source:
            raise ValueError("incorrect unchanged-sensor provenance")
        clipped = item["clipped_fraction"]
        if type(clipped) not in (int, float) or not math.isfinite(clipped) or not 0 <= clipped <= 1:
            raise ValueError("invalid clipping fraction")
        is_noise = condition["corruption"] == "gaussian_noise" and condition["sensor"] == sensor
        seed = item["noise_seed_hex"]
        if is_noise:
            if not isinstance(seed, str) or len(seed) != 32 or any(c not in "0123456789abcdef" for c in seed):
                raise ValueError("invalid deterministic noise seed")
        elif seed is not None:
            raise ValueError("unexpected seed on a non-noise input")
        if frame is not None:
            expected_gt = {key: frame[sensor][key] for key in ("present", "box_xywh", "box_valid")}
            if gt[sensor] != expected_gt or row["attributes"] != frame["attributes"]:
                raise ValueError("cache changed original labels/attributes")


def shard_path(root, condition, sequence):
    for identifier in (condition["id"], sequence):
        if not identifier or identifier in (".", "..") or "/" in identifier or "\\" in identifier:
            raise ValueError("unsafe shard identifier")
    return Path(root) / condition["id"] / sequence


def read_shard(path, condition, context_id, frames=None):
    path = Path(path)
    marker = json.loads((path / "COMPLETED.json").read_text())
    data = path / "rows.jsonl"
    if marker["context_id"] != context_id or marker["sha256"] != sha256_file(data):
        raise RuntimeError(f"corrupt or incompatible shard: {path}")
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)) or keys != marker["keys"] or len(rows) != marker["rows"]:
        raise ValueError("incomplete/duplicate cache keys")
    expected = None if frames is None else {frame["key"]: frame for frame in frames}
    if expected is not None and set(expected) != set(keys):
        raise ValueError("shard membership differs from requested frames")
    for row in rows:
        if row["sequence"] != path.name:
            raise ValueError("row is stored in the wrong sequence shard")
        validate_row(row, condition, context_id, frame=None if expected is None else expected[row["key"]])
    return rows, marker


def write_shard(path, rows, condition, context_id, frames):
    """Publish rows and their completion marker in one atomic directory rename."""
    path = Path(path)
    if path.exists():
        existing, marker = read_shard(path, condition, context_id, frames)
        if existing != rows:
            raise RuntimeError("refusing to replace a different completed shard")
        return marker
    expected = {frame["key"]: frame for frame in frames}
    if not frames:
        raise ValueError("cannot publish an empty shard")
    if len(expected) != len(frames):
        raise ValueError("duplicate expected frame keys")
    if len(rows) != len(expected) or {row["key"] for row in rows} != set(expected):
        raise ValueError("cannot publish an incomplete shard")
    for row in rows:
        validate_row(row, condition, context_id, frame=expected[row["key"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.partial-", dir=path.parent))
    try:
        content = b"".join(encoded(row) for row in rows)
        with (temporary / "rows.jsonl").open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        marker = {"context_id": context_id, "rows": len(rows), "keys": [row["key"] for row in rows],
                  "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        immutable_json(temporary / "COMPLETED.json", marker)
        os.rename(temporary, path)
        return marker
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)  # only the staging directory created by this call


def finalize(root, context_id, frames_by_sequence, condition_records):
    root = Path(root)
    if not frames_by_sequence or not condition_records:
        raise ValueError("cannot finalize an empty cache")
    shards = []
    for condition in condition_records:
        for sequence, frames in sorted(frames_by_sequence.items()):
            path = shard_path(root, condition, sequence)
            _rows, marker = read_shard(path, condition, context_id, frames)
            shards.append({"path": str(path.relative_to(root)), "condition": condition,
                           "rows": marker["rows"], "sha256": marker["sha256"]})
    index = {"schema_version": SCHEMA_VERSION, "context_id": context_id,
             "rows": sum(shard["rows"] for shard in shards), "shards": shards}
    immutable_json(root / "COMPLETED.json", index)
    return index


def iter_cache(root):
    """Validate the whole completed collection before yielding its first row."""
    root = Path(root)
    metadata = json.loads((root / "CACHE.json").read_text())
    index = json.loads((root / "COMPLETED.json").read_text())
    identifier = digest(metadata["context"])
    if identifier != metadata["context_id"] or identifier != index["context_id"]:
        raise ValueError("cache index provenance mismatch")
    total = 0
    seen = set()
    keys_by_condition = {}
    validated = []
    for shard in index["shards"]:
        path = (root / shard["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("cache index points outside its root")
        rows, marker = read_shard(path, shard["condition"], identifier)
        if marker["sha256"] != shard["sha256"] or marker["rows"] != shard["rows"]:
            raise ValueError("cache index/shard mismatch")
        for row in rows:
            key = (row["condition"], row["key"])
            if key in seen:
                raise ValueError("duplicate key across cache shards")
            seen.add(key)
            total += 1
            keys_by_condition.setdefault(row["condition"], []).append(row["key"])
        validated.append((path, shard["condition"]))
    if total != index["rows"]:
        raise ValueError("cache row count mismatch")
    context = metadata["context"]
    if "conditions" in context:
        if set(keys_by_condition) != {condition["id"] for condition in context["conditions"]}:
            raise ValueError("cache condition coverage mismatch")
    if "selected_keys_sha256" in context:
        for keys in keys_by_condition.values():
            if digest(sorted(keys)) != context["selected_keys_sha256"]:
                raise ValueError("cache frame coverage mismatch")
    for path, condition in validated:
        rows, _marker = read_shard(path, condition, identifier)
        yield from rows
