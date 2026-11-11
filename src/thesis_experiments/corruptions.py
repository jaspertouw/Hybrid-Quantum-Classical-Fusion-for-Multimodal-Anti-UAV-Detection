"""Deterministic sensor degradations from the frozen experimental protocol.

Inputs are resized RGB HWC float32 pixels in [0,1], without letterbox padding.
Dropout is represented by None and must bypass detector inference entirely.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Optional

import cv2
import numpy as np

from .detector_contract import EXPECTED_PROTOCOL_SHA256


SENSORS = ("rgb", "thermal")


@dataclass(frozen=True)
class FrameKey:
    split: str
    sequence: str
    frame: int

    def __post_init__(self):
        if self.split not in ("train", "val", "test"):
            raise ValueError("unknown frame split")
        if not self.sequence or any(value in self.sequence for value in ("/", "\\")):
            raise ValueError("sequence must be a single nonempty identifier")
        if type(self.frame) is not int or self.frame < 1:
            raise ValueError("frame must be a positive integer")


@dataclass(frozen=True)
class Condition:
    corruption: str = "clean"
    sensor: Optional[str] = None
    severity: Optional[int] = None

    def __post_init__(self):
        if self.corruption == "clean":
            if self.sensor is not None or self.severity is not None:
                raise ValueError("clean condition has no sensor or severity")
            return
        if self.sensor not in SENSORS:
            raise ValueError("a degradation requires rgb or thermal")
        if self.corruption == "dropout":
            if self.severity is not None:
                raise ValueError("dropout has no severity")
            return
        allowed = {"gaussian_noise", "gaussian_blur", "illumination", "contrast"}
        if self.corruption not in allowed:
            raise ValueError("unknown corruption")
        if type(self.severity) is not int or self.severity not in range(1, 6):
            raise ValueError("severity must be an integer from 1 through 5")
        if self.corruption == "illumination" and self.sensor != "rgb":
            raise ValueError("illumination degradation is RGB-only")
        if self.corruption == "contrast" and self.sensor != "thermal":
            raise ValueError("contrast degradation is thermal-only")

    @property
    def identifier(self):
        if self.corruption == "clean":
            return "clean"
        suffix = "" if self.severity is None else f"_s{self.severity}"
        return f"{self.sensor}_{self.corruption}{suffix}"


@lru_cache(maxsize=1)
def configuration():
    path = Path(__file__).resolve().parents[2] / "experiments/protocol/v1.json"
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("corruptions require the frozen base protocol")
    values = json.loads(content)["corruptions"]
    return MappingProxyType({key: tuple(value) if isinstance(value, list) else value
                             for key, value in values.items()})


def conditions():
    """Clean + six families × five severities + two dropouts = 33 conditions."""
    result = [Condition()]
    for corruption, sensors in (("gaussian_noise", SENSORS), ("gaussian_blur", SENSORS),
                                 ("illumination", ("rgb",)), ("contrast", ("thermal",))):
        result.extend(Condition(corruption, sensor, severity)
                      for sensor in sensors for severity in range(1, 6))
    result.extend(Condition("dropout", sensor) for sensor in SENSORS)
    return tuple(result)


def noise_seed(frame: FrameKey, condition: Condition):
    if condition.corruption != "gaussian_noise":
        raise ValueError("noise seeds are defined only for Gaussian noise")
    identifiers = [configuration()["master_noise_seed"], frame.split, frame.sequence,
                   frame.frame, condition.sensor, condition.corruption, condition.severity]
    content = json.dumps(identifiers, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(content).digest()[:16], "big")


def gaussian_kernel(sigma):
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("blur sigma must be finite and positive")
    radius = math.ceil(3 * sigma)
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(positions * positions) / (2 * sigma * sigma))
    return (kernel / kernel.sum()).astype(np.float32)


@dataclass(frozen=True)
class CorruptionResult:
    image: Optional[np.ndarray]
    clipped_fraction: float = 0.0
    seed: Optional[int] = None


def _validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        raise ValueError("corruption input must be a float32 NumPy image")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("corruption input must have nonempty HWC RGB shape")
    if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
        raise ValueError("corruption input must be finite and in [0,1]")


def apply_to_sensor(image, sensor: str, frame: FrameKey, condition: Condition):
    if sensor not in SENSORS:
        raise ValueError("unknown sensor")
    if condition.corruption == "dropout" and condition.sensor == sensor:
        return CorruptionResult(None)
    _validate_image(image)
    if condition.corruption == "clean" or condition.sensor != sensor:
        return CorruptionResult(image)
    config = configuration()
    level = condition.severity - 1
    seed = None
    if condition.corruption == "gaussian_noise":
        seed = noise_seed(frame, condition)
        generator = np.random.Generator(np.random.PCG64(seed))
        noise = generator.standard_normal(image.shape).astype(np.float32)
        raw = image + np.float32(config["gaussian_noise_sigma"][level]) * noise
    elif condition.corruption == "gaussian_blur":
        kernel = gaussian_kernel(config["gaussian_blur_sigma"][level])
        raw = cv2.sepFilter2D(image, cv2.CV_32F, kernel, kernel,
                             borderType=cv2.BORDER_REFLECT_101)
    elif condition.corruption == "illumination":
        raw = image * np.float32(config["rgb_illumination_multiplier"][level])
    elif condition.corruption == "contrast":
        mean = np.float32(image.mean(dtype=np.float64))
        raw = mean + np.float32(config["thermal_contrast_multiplier"][level]) * (image - mean)
    else:
        raise ValueError("unsupported image corruption")
    clipped_fraction = float(np.mean((raw < 0) | (raw > 1)))
    result = np.ascontiguousarray(np.clip(raw, 0, 1), dtype=np.float32)
    _validate_image(result)
    return CorruptionResult(result, clipped_fraction, seed)


def apply_to_pair(images, frame: FrameKey, condition: Condition):
    if set(images) != set(SENSORS):
        raise ValueError("paired inputs must contain exactly rgb and thermal")
    return {sensor: apply_to_sensor(images[sensor], sensor, frame, condition) for sensor in SENSORS}
