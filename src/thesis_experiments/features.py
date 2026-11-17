"""Shared eight-feature contract; all fitted quantities come from clean training."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np

from .detector_cache import FEATURE_ORDER, digest


def validate_development(rows, *, fitting=False):
    if not rows:
        raise ValueError("empty feature input")
    seen = set()
    for row in rows:
        if row["split"] not in (("train",) if fitting else ("train", "val")):
            raise ValueError("fitting requires train; test is locked")
        if fitting and row["condition"] != "clean":
            raise ValueError("fit shared features on clean training only")
        key = (row["condition"], row["key"])
        if key in seen:
            raise ValueError("duplicate feature row")
        seen.add(key)


def capped_features(rows, caps):
    values = []
    hits = {s: 0 for s in ("rgb", "thermal")}
    for row in rows:
        value = []
        for sensor in ("rgb", "thermal"):
            candidate = row[sensor]["candidate"]
            if candidate is None:
                value.extend((0., 1., 0., caps[sensor]))
            else:
                spatial = candidate["spatial_uncertainty_corrected"]
                hits[sensor] += int(spatial > caps[sensor])
                value.extend((candidate["beliefs"][0], candidate["semantic_uncertainty"],
                              candidate["objectness"], min(spatial, caps[sensor])))
        values.append(value)
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite feature values")
    return result, hits


def fit_features(rows):
    validate_development(rows, fitting=True)
    caps = {}
    for sensor in ("rgb", "thermal"):
        values = [r[sensor]["candidate"]["spatial_uncertainty_corrected"]
                  for r in rows if r[sensor]["candidate"] is not None]
        if not values:
            raise ValueError(f"no training candidates for {sensor}")
        caps[sensor] = float(np.percentile(values, 99.5))
    raw, _ = capped_features(rows, caps)
    return {"schema_version": 1, "feature_order": list(FEATURE_ORDER), "caps": caps,
            "minimum": raw.min(axis=0).tolist(), "maximum": raw.max(axis=0).tolist(),
            "fit_split": "train", "fit_rows": len(rows),
            "fit_keys_sha256": digest(sorted(r["key"] for r in rows)),
            "fit_context_ids": sorted({r["context_id"] for r in rows})}


def transform_features(rows, fitted):
    validate_development(rows)
    if fitted["fit_split"] != "train" or fitted["feature_order"] != list(FEATURE_ORDER):
        raise ValueError("invalid fitted feature contract")
    raw, hits = capped_features(rows, fitted["caps"])
    minimum, maximum = np.array(fitted["minimum"]), np.array(fitted["maximum"])
    normalized = np.clip((raw - minimum) / np.maximum(maximum - minimum, 1e-8), 0, 1)
    normalized[:, maximum == minimum] = 0
    return raw, normalized, {"rows": len(rows), "spatial_cap_hits": hits,
                             "below_training_min": (raw < minimum).sum(axis=0).tolist(),
                             "above_training_max": (raw > maximum).sum(axis=0).tolist()}


def training_sample(rows, cap=20000, seed=11):
    """Largest-remainder allocation preserves sequence/class fractions under a cap."""
    validate_development(rows, fitting=True)
    if cap < 1:
        raise ValueError("training cap must be positive")
    if len(rows) <= cap:
        return np.arange(len(rows), dtype=np.int64)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["sequence"], row["paired_presence"])].append(index)
    fraction = cap / len(rows)
    allocation = {key: int(len(indices) * fraction) for key, indices in groups.items()}
    order = sorted(groups, key=lambda key: (-(len(groups[key]) * fraction - allocation[key]), key))
    for key in order[:cap - sum(allocation.values())]:
        allocation[key] += 1
    selected = []
    for key, indices in groups.items():
        ranked = sorted(indices, key=lambda i: hashlib.sha256(
            json.dumps([seed, rows[i]["key"]], separators=(",", ":")).encode()).digest())
        selected.extend(ranked[:allocation[key]])
    return np.asarray(sorted(selected, key=lambda i: rows[i]["key"]), dtype=np.int64)


def validation_sample(rows, cap=2000, seed=11):
    """Prediction-independent equal-fraction sequence sampling for runtime pilots."""
    validate_development(rows)
    if any(r["split"] != "val" or r["condition"] != "clean" for r in rows):
        raise ValueError("validation sampling requires clean validation")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["sequence"]].append(i)
    cap = min(cap, len(rows))
    allocation = {s: int(len(v) * cap / len(rows)) for s, v in groups.items()}
    ranked = sorted(groups, key=lambda s: (-(len(groups[s]) * cap / len(rows) - allocation[s]), s))
    for s in ranked[:cap - sum(allocation.values())]:
        allocation[s] += 1
    chosen = []
    for s, indices in groups.items():
        chosen.extend(sorted(indices, key=lambda i: hashlib.sha256(
            f"validation-runtime:{seed}:{rows[i]['key']}".encode()).digest())[:allocation[s]])
    return np.array(sorted(chosen, key=lambda i: rows[i]["key"]), dtype=np.int64)
