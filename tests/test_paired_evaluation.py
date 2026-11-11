import pytest

from src.thesis_experiments.paired_evaluation import (
    Candidate,
    aggregate_frame_weighted,
    corrected_dms_winner,
    evaluate_legacy_dms_sequence,
    evaluate_sequence,
    legacy_dms_score,
    legacy_dms_winner,
)


def frame(*, thermal=None, rgb=None):
    def sensor(box):
        return {"present": box is not None, "box_valid": box is not None, "box_xywh": box}

    return {
        "paired_presence": thermal is not None or rgb is not None,
        "thermal": sensor(thermal),
        "rgb": sensor(rgb),
    }


def test_methodology_edtc_uses_paired_presence_and_sensor_specific_box():
    frames = [frame(rgb=[0, 0, 10, 10])]
    perfect = Candidate("rgb", (0, 0, 10, 10))
    assert evaluate_sequence(frames, [perfect]).accuracy == pytest.approx(1.0)
    assert evaluate_legacy_dms_sequence(frames, [perfect]).accuracy == pytest.approx(0.0)
    assert evaluate_sequence(frames, [None]).accuracy == pytest.approx(-0.2)
    assert evaluate_legacy_dms_sequence(frames, [None]).accuracy == pytest.approx(1.0)


def test_methodology_penalty_uses_present_denominator():
    frames = [
        frame(thermal=[0, 0, 10, 10]),
        frame(thermal=[0, 0, 10, 10]),
        frame(),
        frame(),
    ]
    result = evaluate_sequence(frames, [None] * 4)
    assert result.accuracy == pytest.approx(0.3)
    assert evaluate_legacy_dms_sequence(frames, [None] * 4).accuracy == pytest.approx(
        0.337549521, abs=1e-9
    )


def test_invalid_positive_box_is_present_but_zero_overlap():
    invalid = {
        "paired_presence": True,
        "thermal": {"present": True, "box_valid": False, "box_xywh": [1, 1, 0, 4]},
        "rgb": {"present": False, "box_valid": False, "box_xywh": None},
    }
    result = evaluate_sequence([invalid], [Candidate("thermal", (1, 1, 2, 4))])
    assert result.failures == 1
    assert result.accuracy == pytest.approx(-0.2)


def test_absent_false_positive_receives_no_absence_reward():
    result = evaluate_sequence([frame()], [Candidate("rgb", (1, 1, 2, 2))])
    assert result.accuracy == pytest.approx(0.0)


def test_corrected_and_legacy_dms_tie_rules_are_explicit():
    assert corrected_dms_winner(0.1, 0.1) == "thermal"
    assert legacy_dms_winner(0.1, 0.1) == "rgb"
    assert corrected_dms_winner(None, 0.2) == "thermal"
    assert legacy_dms_winner(0.2, None) == "rgb"


def test_legacy_dms_score_reproduces_first_power_native_normalization():
    assert legacy_dms_score([2, 4, 6, 8], [0, 0, 2, 4]) == pytest.approx(1.75)


def test_frame_weighted_pooling_uses_global_present_count():
    first = evaluate_sequence([frame(thermal=[0, 0, 2, 2])], [None])
    second = evaluate_sequence([frame(), frame()], [None, None])
    pooled = aggregate_frame_weighted([first, second])
    assert pooled.frames == 3
    assert pooled.present == 1
    assert pooled.accuracy == pytest.approx(2 / 3 - 0.2)
