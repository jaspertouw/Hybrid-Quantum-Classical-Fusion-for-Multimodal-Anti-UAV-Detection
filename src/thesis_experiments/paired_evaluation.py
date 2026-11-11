"""Paired RGB--thermal box selection and EDTC evaluation contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence


SENSORS = frozenset(("rgb", "thermal"))


@dataclass(frozen=True)
class Candidate:
    """One sensor-native candidate before or after an acceptance decision."""

    sensor: str
    box_xywh: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if self.sensor not in SENSORS:
            raise ValueError(f"unknown candidate sensor: {self.sensor}")
        if len(self.box_xywh) != 4 or not all(math.isfinite(v) for v in self.box_xywh):
            raise ValueError("candidate box must contain four finite values")
        if self.box_xywh[2] <= 0.0 or self.box_xywh[3] <= 0.0:
            raise ValueError("candidate box must have positive width and height")


@dataclass(frozen=True)
class EdtcResult:
    frames: int
    present: int
    absent: int
    reward_sum: float
    failures: int
    accuracy: float


def iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    """Continuous-coordinate IoU for positive-width ``[x, y, w, h]`` boxes."""
    if len(first) != 4 or len(second) != 4:
        raise ValueError("IoU boxes must have four coordinates")
    if not all(math.isfinite(float(value)) for value in (*first, *second)):
        raise ValueError("IoU boxes must be finite")
    ax, ay, aw, ah = map(float, first)
    bx, by, bw, bh = map(float, second)
    if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
        return 0.0
    intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def _ground_truth(frame: Mapping[str, object], sensor: str) -> Optional[Sequence[float]]:
    sensor_record = frame.get(sensor)
    if not isinstance(sensor_record, Mapping):
        raise ValueError(f"frame lacks {sensor} record")
    if not sensor_record.get("present", False):
        return None
    box = sensor_record.get("box_xywh")
    if not sensor_record.get("box_valid", False) or not isinstance(box, list):
        return None
    return box


def frame_iou(frame: Mapping[str, object], candidate: Optional[Candidate]) -> float:
    """Match a candidate only against ground truth from its own sensor."""
    if candidate is None:
        return 0.0
    ground_truth = _ground_truth(frame, candidate.sensor)
    return iou_xywh(candidate.box_xywh, ground_truth) if ground_truth is not None else 0.0


def evaluate_sequence(
    frames: Sequence[Mapping[str, object]],
    candidates: Sequence[Optional[Candidate]],
    *,
    alpha: float = 0.2,
    beta: float = 0.3,
) -> EdtcResult:
    """Evaluate the thesis EDTC definition for one paired sequence."""
    if len(frames) != len(candidates):
        raise ValueError("frame and candidate counts differ")
    if not frames:
        raise ValueError("cannot evaluate an empty sequence")
    if alpha < 0.0 or beta <= 0.0:
        raise ValueError("invalid EDTC penalty parameters")

    reward_sum = 0.0
    failures = 0
    present = 0
    for frame, candidate in zip(frames, candidates):
        paired_presence = frame.get("paired_presence")
        if not isinstance(paired_presence, bool):
            raise ValueError("frame lacks Boolean paired_presence")
        if paired_presence:
            present += 1
            overlap = frame_iou(frame, candidate)
            reward_sum += overlap
            failures += int(overlap <= 0.0)
        else:
            reward_sum += float(candidate is None)

    frame_count = len(frames)
    penalty = alpha * ((failures / present) ** beta) if present else 0.0
    return EdtcResult(
        frames=frame_count,
        present=present,
        absent=frame_count - present,
        reward_sum=reward_sum,
        failures=failures,
        accuracy=reward_sum / frame_count - penalty,
    )


def aggregate_frame_weighted(results: Iterable[EdtcResult]) -> float:
    """Weight each sequence's EDTC score by its number of frames."""
    values = tuple(results)
    if not values or sum(result.frames for result in values) <= 0:
        raise ValueError("no non-empty sequence results")
    return sum(result.frames * result.accuracy for result in values) / sum(
        result.frames for result in values
    )


def aggregate_pooled_components(results: Iterable[EdtcResult], *, alpha=0.2, beta=0.3) -> EdtcResult:
    """Legacy audit diagnostic: pool counts before applying the nonlinear penalty."""
    values = tuple(results)
    if not values:
        raise ValueError("no sequence results")
    frames = sum(result.frames for result in values)
    present = sum(result.present for result in values)
    reward = sum(result.reward_sum for result in values)
    failures = sum(result.failures for result in values)
    penalty = alpha * ((failures / present) ** beta) if present else 0.0
    return EdtcResult(
        frames=frames,
        present=present,
        absent=frames - present,
        reward_sum=reward,
        failures=failures,
        accuracy=reward / frames - penalty,
    )


def legacy_native_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    """Released runner's rounded endpoints and inclusive (+1) dimensions.

    Python round and torch.round both use ties-to-even. Input coordinates have
    already been scaled/clipped to the sensor's native image bounds.
    """
    Candidate("rgb", tuple(box))  # validate geometry before conversion
    x, y, width, height = box
    x1, y1, x2, y2 = (float(round(v)) for v in (x, y, x + width, y + height))
    return x1, y1, x2 - x1 + 1.0, y2 - y1 + 1.0


def corrected_dms_winner(
    rgb_spatial_uncertainty: Optional[float],
    thermal_spatial_uncertainty: Optional[float],
) -> Optional[str]:
    """Choose the lowest corrected uncertainty, with thermal winning ties."""
    values = {"rgb": rgb_spatial_uncertainty, "thermal": thermal_spatial_uncertainty}
    for sensor, value in values.items():
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"invalid {sensor} spatial uncertainty")
    if rgb_spatial_uncertainty is None:
        return "thermal" if thermal_spatial_uncertainty is not None else None
    if thermal_spatial_uncertainty is None:
        return "rgb"
    return "thermal" if thermal_spatial_uncertainty <= rgb_spatial_uncertainty else "rgb"


def legacy_dms_score(raw_grid_variance: Sequence[float], native_box_xywh: Sequence[float]) -> float:
    """Reproduce the released DMS scalar: raw grid variance / native dimension."""
    if len(raw_grid_variance) != 4 or len(native_box_xywh) != 4:
        raise ValueError("legacy DMS inputs must each contain four values")
    width, height = float(native_box_xywh[2]), float(native_box_xywh[3])
    if width <= 0.0 or height <= 0.0:
        raise ValueError("legacy DMS box dimensions must be positive")
    values = tuple(float(value) for value in raw_grid_variance)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("legacy DMS variances must be finite and positive")
    denominators = (width, height, width, height)
    return sum(value / denominator for value, denominator in zip(values, denominators)) / 4.0


def legacy_dms_winner(
    rgb_score: Optional[float], thermal_score: Optional[float]
) -> Optional[str]:
    """Reproduce the released winner rule, including its RGB tie preference."""
    if rgb_score is None:
        return "thermal" if thermal_score is not None else None
    if thermal_score is None:
        return "rgb"
    if not all(math.isfinite(value) and value >= 0.0 for value in (rgb_score, thermal_score)):
        raise ValueError("invalid legacy DMS score")
    return "rgb" if rgb_score <= thermal_score else "thermal"


def evaluate_legacy_dms_sequence(
    frames: Sequence[Mapping[str, object]],
    candidates: Sequence[Optional[Candidate]],
    *,
    alpha: float = 0.2,
    beta: float = 0.3,
    failure_iou_threshold: float = 1e-5,
) -> EdtcResult:
    """Reproduce the released paired evaluator for an explicit audit only."""
    if len(frames) != len(candidates) or not frames:
        raise ValueError("invalid legacy evaluator inputs")
    reward_sum = 0.0
    failures = 0
    thermal_present_count = 0
    for frame, candidate in zip(frames, candidates):
        thermal = frame.get("thermal")
        if not isinstance(thermal, Mapping):
            raise ValueError("frame lacks thermal record")
        thermal_present = bool(thermal.get("present", False))
        if thermal_present:
            thermal_present_count += 1
            overlap = frame_iou(frame, candidate)
            # Released iou_xywh adds 1e-8 to the union denominator.
            if candidate is not None:
                gt = _ground_truth(frame, candidate.sensor)
                if gt is not None and overlap > 0:
                    box = candidate.box_xywh
                    intersection = max(0.0, min(box[0] + box[2], gt[0] + gt[2]) - max(box[0], gt[0])) * max(
                        0.0, min(box[1] + box[3], gt[1] + gt[3]) - max(box[1], gt[1])
                    )
                    union = box[2] * box[3] + gt[2] * gt[3] - intersection
                    overlap = intersection / (union + 1e-8)
            reward_sum += overlap
            failures += int(overlap < failure_iou_threshold)
        else:
            reward_sum += float(candidate is None)
    frame_count = len(frames)
    penalty = alpha * ((failures / frame_count) ** beta)
    return EdtcResult(
        frames=frame_count,
        present=thermal_present_count,
        absent=frame_count - thermal_present_count,
        reward_sum=reward_sum,
        failures=failures,
        accuracy=reward_sum / frame_count - penalty,
    )
