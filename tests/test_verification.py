import copy

import numpy as np
import pytest

pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

from test_shared_fusion import row
from src.thesis_experiments.metrics import bootstrap_edtc, evaluation_data, oracle_acceptance, robustness_summary, sequence_scores
from src.thesis_experiments.io import atomic_write_json
from src.thesis_experiments.neural_fusion import NeuralFusion, train_neural
from src.thesis_experiments.paired_evaluation import Candidate, evaluate_sequence
from src.thesis_experiments.verification import synthetic_run, verify_quantum_reference


def test_fresh_synthetic_cache_to_export_runs_are_reproducible(tmp_path):
    torch.set_num_threads(2)
    first, p = synthetic_run(tmp_path / "first")
    second, q = synthetic_run(tmp_path / "second")
    assert first == second
    atomic_write_json(tmp_path / "first_results.json", first)
    atomic_write_json(tmp_path / "second_results.json", second)
    assert (tmp_path / "first_results.json").read_bytes() == (tmp_path / "second_results.json").read_bytes()
    for name in p:
        np.testing.assert_array_equal(p[name], q[name])


def test_independent_dense_quantum_reference_matches_all_layouts():
    result = verify_quantum_reference()
    assert result["cases"] == 36
    assert result["maximum_absolute_error"] < 1e-12


def test_vectorized_edtc_matches_original_evaluator_and_fixed_box_oracle():
    rows = [row(1, split="val"), row(2, split="val", present=False),
            row(3, sequence="other", split="val", missing=True), row(4, sequence="other", split="val")]
    rows[-1]["ground_truth"]["thermal"]["box_valid"] = False
    data = evaluation_data(rows)
    p = np.array([0.9, 0.8, 0.4, 0.9])
    scores, _ = sequence_scores(data, p)
    direct = []
    for sequence in data["sequences"]:
        frames, boxes = [], []
        for i, r in enumerate(rows):
            if r["sequence"] != sequence:
                continue
            frames.append({"paired_presence": r["paired_presence"], **r["ground_truth"]})
            candidate = r["thermal"]["candidate"]
            boxes.append(Candidate("thermal", tuple(candidate["box_xywh_native"]))
                         if candidate is not None and p[i] > 0.5 else None)
        direct.append(evaluate_sequence(frames, boxes).accuracy)
    np.testing.assert_allclose(scores, direct, atol=1e-15)
    bound = oracle_acceptance(data)
    assert bound["edtc_sequence_mean"] >= scores.mean()
    assert bound["annotation_assisted"]


def test_robustness_averages_conditions_not_frames_and_exports_deltas():
    conditions = {"clean": {"brier": 0.2, "edtc_sequence_mean": 0.8},
                  **{f"c{i}": {"brier": 0.3, "edtc_sequence_mean": 0.7} for i in range(30)},
                  "rgb_dropout": {"brier": 0.9, "edtc_sequence_mean": 0.1}}
    result = robustness_summary(conditions, [f"c{i}" for i in range(30)])
    assert result["metrics"]["edtc_sequence_mean"]["mean"] == pytest.approx(0.7)
    assert result["metrics"]["brier"]["delta_by_condition"]["c0"] == pytest.approx(0.1)
    assert result["metrics"]["ap50"]["supported_conditions"] == 0
    del conditions["c1"]
    with pytest.raises(ValueError, match="coverage"):
        robustness_summary(conditions, [f"c{i}" for i in range(30)])


def test_bootstrap_keeps_conditions_paired_and_averages_fixed_seeds():
    first = {11: {"c0": {"a": 0.6, "b": 0.4}, "c1": {"a": 0.4, "b": 0.6}}}
    second = {11: {"c0": {"a": 0.5, "b": 0.5}, "c1": {"a": 0.5, "b": 0.5}}}
    first[22], second[22] = copy.deepcopy(first[11]), copy.deepcopy(second[11])
    result = bootstrap_edtc(first, second, ["c0", "c1"], replicates=2000)
    assert result["interval95"] == pytest.approx([0, 0])
    assert result["undefined_replicates"] == 0
    del first[22]["c1"]["b"]
    with pytest.raises(ValueError, match="coverage"):
        bootstrap_edtc(first, second, ["c0", "c1"])


def test_training_records_best_epoch_and_actual_stop_reason(monkeypatch):
    model = NeuralFusion("mlp")
    x = np.zeros((4, 8))
    labels = np.array([True, False, True, False])
    data = evaluation_data([row(i, split="val", present=bool(labels[i - 1])) for i in range(1, 5)])
    monkeypatch.setattr(model, "predict", lambda _: (np.full(4, 0.7), np.full(4, 0.5)))
    result = train_neural(model, x, labels, x, data, epochs=5, patience=1)
    assert result["best_epoch"] == 1
    assert result["epochs"] == 2
    assert result["stop_reason"] == "patience"
    with pytest.raises(ValueError, match="training"):
        train_neural(model, x, labels, x, data, epochs=0)
