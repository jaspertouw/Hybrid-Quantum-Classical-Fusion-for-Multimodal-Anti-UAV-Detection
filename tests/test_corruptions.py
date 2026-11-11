import numpy as np
import pytest

from src.thesis_experiments.corruptions import (
    Condition, FrameKey, apply_to_pair, apply_to_sensor, conditions, gaussian_kernel, noise_seed,
)
from src.thesis_experiments import degraded_inputs


FRAME = FrameKey("val", "sequence", 1)


def test_condition_grid_has_exactly_33_distinct_conditions():
    grid = conditions()
    assert len(grid) == len({condition.identifier for condition in grid}) == 33
    assert sum(condition.corruption == "gaussian_noise" for condition in grid) == 10
    assert sum(condition.corruption == "gaussian_blur" for condition in grid) == 10
    assert sum(condition.corruption == "dropout" for condition in grid) == 2


@pytest.mark.parametrize("condition", conditions())
def test_all_conditions_preserve_other_sensor_and_input_data(condition):
    original = np.linspace(0, 1, 9 * 11 * 3, dtype=np.float32).reshape(9, 11, 3)
    images = {"rgb": original.copy(), "thermal": original.copy()}
    result = apply_to_pair(images, FRAME, condition)
    for sensor in images:
        np.testing.assert_array_equal(images[sensor], original)
        if condition.sensor != sensor:
            assert result[sensor].image is images[sensor]
        if condition.corruption == "dropout" and condition.sensor == sensor:
            assert result[sensor].image is None
        else:
            pixels = result[sensor].image
            assert pixels.dtype == np.float32 and pixels.shape == original.shape
            assert np.isfinite(pixels).all() and pixels.min() >= 0 and pixels.max() <= 1


def test_noise_is_keyed_order_independent_and_does_not_touch_global_rng():
    image = np.full((64, 64, 3), 0.5, dtype=np.float32)
    condition = Condition("gaussian_noise", "rgb", 2)
    np.random.seed(123)
    expected_global = np.random.random(5)
    np.random.seed(123)
    first = apply_to_sensor(image, "rgb", FRAME, condition)
    apply_to_sensor(image, "rgb", FrameKey("val", "other", 5), condition)
    second = apply_to_sensor(image, "rgb", FRAME, condition)
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(np.random.random(5), expected_global)
    seeds = {
        noise_seed(FRAME, condition),
        noise_seed(FrameKey("train", "sequence", 1), condition),
        noise_seed(FrameKey("val", "other", 1), condition),
        noise_seed(FrameKey("val", "sequence", 2), condition),
        noise_seed(FRAME, Condition("gaussian_noise", "thermal", 2)),
        noise_seed(FRAME, Condition("gaussian_noise", "rgb", 3)),
    }
    assert len(seeds) == 6


def test_noise_scale_and_clipping_are_measured():
    image = np.full((100, 200, 3), 0.5, dtype=np.float32)
    result = apply_to_sensor(image, "rgb", FRAME, Condition("gaussian_noise", "rgb", 1))
    delta = result.image - image
    assert abs(float(delta.mean())) < 0.0003
    assert float(delta.std()) == pytest.approx(0.02, abs=0.0003)
    extremes = np.zeros((100, 100, 3), dtype=np.float32)
    clipped = apply_to_sensor(extremes, "rgb", FRAME, Condition("gaussian_noise", "rgb", 5))
    assert 0.47 < clipped.clipped_fraction < 0.53


def test_noise_is_identical_under_parallel_scheduling():
    from concurrent.futures import ThreadPoolExecutor

    image = np.full((8, 9, 3), 0.5, dtype=np.float32)
    keys = [FrameKey("val", "sequence", index) for index in range(1, 9)]
    condition = Condition("gaussian_noise", "rgb", 3)

    def run(key):
        return apply_to_sensor(image, "rgb", key, condition).image

    serial = {key: run(key) for key in keys}
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(run, reversed(keys)))
    for key, pixels in zip(reversed(keys), actual):
        np.testing.assert_array_equal(serial[key], pixels)


def test_blur_matches_independent_reflection_convolution():
    image = np.zeros((5, 6, 3), dtype=np.float32)
    image[0, 0, 0] = 1
    kernel = gaussian_kernel(1.0)
    assert len(kernel) == 7
    reference_kernel = np.exp(-np.arange(-3, 4, dtype=np.float64)**2 / 2)
    reference_kernel /= reference_kernel.sum()
    padded = np.pad(image, ((3, 3), (3, 3), (0, 0)), mode="reflect")
    reference = np.empty_like(image)
    weights = np.outer(reference_kernel, reference_kernel)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            reference[y, x] = (padded[y:y + 7, x:x + 7] * weights[:, :, None]).sum(axis=(0, 1))
    actual = apply_to_sensor(image, "rgb", FRAME, Condition("gaussian_blur", "rgb", 1)).image
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-7)
    assert np.count_nonzero(actual[:, :, 1:]) == 0
    assert [len(gaussian_kernel(sigma)) for sigma in (1, 2, 3, 4, 5)] == [7, 13, 19, 25, 31]


def test_photometric_operators_match_methodology():
    image = np.array([0.2, 0.4, 0.6], dtype=np.float32).reshape(1, 1, 3)
    dark = apply_to_sensor(image, "rgb", FRAME, Condition("illumination", "rgb", 5)).image
    np.testing.assert_allclose(dark, image * 0.1)
    contrast = apply_to_sensor(image, "thermal", FRAME, Condition("contrast", "thermal", 5)).image
    np.testing.assert_allclose(contrast, 0.4 + 0.1 * (image - 0.4))


def test_input_adapter_keeps_letterbox_padding_and_skips_dropout_decode(monkeypatch):
    chw = np.full((3, 6, 8), 114, dtype=np.uint8)
    chw[:, 1:5, 1:7] = 200
    calls = []

    def prepare(path, imgsz, stride):
        calls.append(path)
        return chw, str(path), ((4, 6), ((1, 1), (1, 1)))

    monkeypatch.setattr(degraded_inputs, "prepare_image", prepare)
    omitted = degraded_inputs.prepare_conditioned_sensor(
        "not_read", "rgb", FRAME, Condition("dropout", "rgb"))
    assert omitted is None and not calls
    output = degraded_inputs.prepare_conditioned_sensor(
        "image", "rgb", FRAME, Condition("illumination", "rgb", 5))
    assert output.pixels_chw.shape == chw.shape
    np.testing.assert_allclose(output.pixels_chw[:, 0, :], np.float32(114 / 255))
    np.testing.assert_allclose(output.pixels_chw[:, 1:5, 1:7], np.float32(200 / 255) * 0.1)
    np.testing.assert_array_equal(chw[:, 1:5, 1:7], 200)
    clean = degraded_inputs.prepare_conditioned_sensor("image", "rgb", FRAME, Condition())
    np.testing.assert_array_equal(clean.pixels_chw, chw.astype(np.float32) / np.float32(255))


@pytest.mark.parametrize("kwargs", [
    {"corruption": "contrast", "sensor": "rgb", "severity": 1},
    {"corruption": "illumination", "sensor": "thermal", "severity": 1},
    {"corruption": "gaussian_noise", "sensor": "rgb", "severity": 0},
    {"corruption": "dropout", "sensor": "rgb", "severity": 1},
])
def test_invalid_conditions_fail(kwargs):
    with pytest.raises(ValueError):
        Condition(**kwargs)


def test_frozen_corruption_specification_matches_validated_sources():
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    specification = json.loads((root / "experiments/protocol/corruptions_v1.json").read_text())
    assert specification["status"] == "frozen"
    for relative, expected in specification["implementation_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
