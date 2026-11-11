"""Attach Phase 5 operators to frozen Phase 2 preparation without modifying it."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .corruptions import Condition, FrameKey, apply_to_sensor
from .corruptions import SENSORS
from .detector_preprocessing import prepare_image


@dataclass(frozen=True)
class PreparedInput:
    pixels_chw: np.ndarray  # float32 [0,1], ready for torch.from_numpy (no second /255)
    geometry: tuple
    clipped_fraction: float
    noise_seed_hex: Optional[str]


def prepare_conditioned_sensor(path, sensor: str, frame: FrameKey, condition: Condition,
                               *, imgsz=640, stride=32):
    if sensor not in SENSORS:
        raise ValueError("unknown sensor")
    if condition.corruption == "dropout" and condition.sensor == sensor:
        return None  # do not decode the omitted image or call its model
    chw, _path, geometry = prepare_image(path, imgsz, stride)
    pixels = chw.astype(np.float32) / np.float32(255)
    if condition.corruption == "clean" or condition.sensor != sensor:
        return PreparedInput(pixels, geometry, 0.0, None)
    _original_shape, (_ratio, (pad_x, pad_y)) = geometry
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    content = pixels[:, top:pixels.shape[1] - bottom, left:pixels.shape[2] - right]
    result = apply_to_sensor(content.transpose(1, 2, 0), sensor, frame, condition)
    output = pixels.copy()
    output[:, top:pixels.shape[1] - bottom, left:pixels.shape[2] - right] = result.image.transpose(2, 0, 1)
    return PreparedInput(np.ascontiguousarray(output), geometry, result.clipped_fraction,
                         None if result.seed is None else f"{result.seed:032x}")
