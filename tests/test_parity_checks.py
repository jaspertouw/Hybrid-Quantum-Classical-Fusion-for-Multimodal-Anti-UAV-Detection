import pytest

from src.thesis_experiments.parity_checks import compare_values


def test_parity_reports_magnitude_without_relaxing_tolerance():
    result = compare_values([0.5], [0.501])
    assert result["passed"] is False
    assert result["max_absolute_error"] == pytest.approx(0.001)
    assert result["max_tolerance_ratio"] > 1


def test_parity_accepts_small_errors_and_checks_near_zero():
    assert compare_values([0.5], [0.500001])["passed"] is True
    assert compare_values([0], [0.00002])["passed"] is False


def test_parity_rejects_nan_and_different_lengths():
    with pytest.raises(ValueError, match="non-finite"):
        compare_values([float("nan")], [1])
    with pytest.raises(ValueError, match="equal non-zero"):
        compare_values([1], [1, 2])
