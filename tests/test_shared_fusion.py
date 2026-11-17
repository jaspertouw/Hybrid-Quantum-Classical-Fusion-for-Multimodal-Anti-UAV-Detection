import copy

import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.thesis_experiments.features import fit_features, training_sample, transform_features
from src.thesis_experiments.fusion import abf, disagreement, predict_reference
from src.thesis_experiments.metrics import average_precision, evaluate, evaluation_data, paired_bootstrap, select_threshold
from src.thesis_experiments.models import ClassicalModel


def row(number=1, *, present=True, sequence="seq", split="train", missing=False):
    proposal = {"beliefs": [0.6, 0.1], "semantic_uncertainty": 0.3, "objectness": 0.8,
                "spatial_uncertainty_corrected": float(number), "box_xywh_native": [0, 0, 10, 10]}
    return {"key": f"{split}/{sequence}/{number:06d}", "sequence": sequence, "condition": "clean",
            "split": split, "context_id": "fixture", "paired_presence": present, "attributes": ["LI"],
            "ground_truth": {s: {"present": present, "box_valid": present,
                                   "box_xywh": [0, 0, 10, 10] if present else None} for s in ("rgb", "thermal")},
            **{s: {"candidate": None if missing else copy.deepcopy(proposal)} for s in ("rgb", "thermal")}}


def test_train_only_caps_missing_encoding_and_constant_features():
    rows = [row(1), row(100)]
    fitted = fit_features(rows)
    assert fitted["caps"]["rgb"] == pytest.approx(99.505)
    sample = row(2, split="val", missing=True)
    raw, normalized, report = transform_features([sample], fitted)
    np.testing.assert_array_equal(raw[0, :4], [0, 1, 0, fitted["caps"]["rgb"]])
    assert normalized[0, 0] == 0  # constant training feature maps to zero
    assert normalized[0, 3] == 1
    assert report["below_training_min"][0] == 1
    with pytest.raises(ValueError, match="train"):
        fit_features([sample])
    sample["split"] = "test"
    with pytest.raises(ValueError, match="locked"):
        transform_features([sample], fitted)


def test_features_ignore_labels_and_coordinates_and_do_not_mutate_fit():
    fitted = fit_features([row(1), row(2, missing=True)])
    sample = row(3, split="val")
    changed = copy.deepcopy(sample)
    changed["ground_truth"] = None
    changed["paired_presence"] = False
    for sensor in ("rgb", "thermal"):
        del changed[sensor]["candidate"]["box_xywh_native"]
    before = copy.deepcopy(fitted)
    np.testing.assert_array_equal(transform_features([sample], fitted)[1], transform_features([changed], fitted)[1])
    assert fitted == before


def test_sampling_exact_cap_order_independence_and_class_support():
    rows = [row(i, sequence=str(s), present=i % 2 == 0) for s in range(4) for i in range(1, 21)]
    chosen = training_sample(rows, cap=16)
    keys = {rows[i]["key"] for i in chosen}
    reversed_rows = list(reversed(rows))
    assert keys == {reversed_rows[i]["key"] for i in training_sample(reversed_rows, cap=16)}
    assert len(keys) == 16
    assert sum(rows[i]["paired_presence"] for i in chosen) == 8


def test_missing_and_conflicting_reference_opinions():
    r = row()
    r["rgb"]["candidate"] = None
    p, u = predict_reference([r], "abf")
    assert p[0] == pytest.approx(0.6 / 1.3 + (0.6 / 1.3) / 2)
    assert u[0] == pytest.approx(0.6 / 1.3)
    conflicting = row()
    conflicting["rgb"]["candidate"].update(beliefs=[0.9, 0], semantic_uncertainty=0.1)
    conflicting["thermal"]["candidate"].update(beliefs=[0, 0.9], semantic_uncertainty=0.1)
    _, plain_u = predict_reference([conflicting], "abf")
    _, discounted_u = predict_reference([conflicting], "dbf")
    assert discounted_u[0] > plain_u[0]
    assert disagreement([conflicting])["inherited_conflict"][0] == pytest.approx(0.81)
    b, zero_u = abf(np.array([[1., 0]]), np.array([0.]), np.array([[0., 1]]), np.array([0.]))
    np.testing.assert_array_equal(b, [[0.5, 0.5]])
    assert zero_u[0] == 0


def test_hand_calculated_edtc_brier_ece_and_acceptance():
    data = evaluation_data([row(1, split="val"), row(2, split="val", present=False)])
    result = evaluate(data, [0.9, 0.8])
    assert result["edtc_sequence_mean"] == 0.5
    assert result["brier"] == pytest.approx(0.325)
    assert result["ece"] == pytest.approx(0.45)
    assert result["ap50"] == result["map50_95"] == 1
    assert result["specificity"] == 0
    assert select_threshold(data, [0.9, 0.8]) == pytest.approx(0.85)
    reference = evaluate(data, [1, 1], box_scores=False)
    assert "edtc_sequence_mean" not in reference


def test_ap_includes_missing_positives_and_uses_stable_frame_ties():
    data = evaluation_data([row(1, split="val"), row(2, split="val", missing=True)])
    assert average_precision(data, [0.9, 0.1], 0.5) == pytest.approx(51 / 101)
    a, b = row(1, split="val", present=False), row(2, split="val")
    data = evaluation_data([b, a])
    assert average_precision(data, [0.5, 0.5], 0.5) == 0.5


def test_undefined_ranking_and_sensor_specific_ground_truth():
    r = row(1, split="val")
    r["ground_truth"]["thermal"] = {"present": False, "box_valid": False, "box_xywh": None}
    data = evaluation_data([r])  # thermal tie winner has no matching annotation
    result = evaluate(data, [0.9], [0.1])
    assert result["edtc_sequence_mean"] == -0.2
    assert result["presence_error_ranking"]["confidence"]["auroc"] is None
    assert result["specificity"] is None
    assert result["post_acceptance_miou"] == 0
    rgb = evaluate(evaluation_data([r], sensor="rgb"), [0.9])
    assert rgb["edtc_sequence_mean"] == 1


def test_bootstrap_requires_paired_sequences_and_constant_delta_is_exact():
    result = paired_bootstrap({"a": 0.7, "b": 0.5}, {"a": 0.6, "b": 0.4}, replicates=100)
    assert result["interval95"] == pytest.approx([0.1, 0.1])
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap({"a": 1}, {"b": 1})


@pytest.mark.parametrize("kind,settings", [("svm", {"C": 1., "gamma": 0.1}), ("forest", {"max_depth": 4, "min_samples_leaf": 1})])
def test_classical_models_group_calibration_and_roundtrip(tmp_path, kind, settings):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(120, 8))
    y = np.tile([False, True], 60)
    x[:, 0] = y + rng.normal(scale=0.05, size=120)
    groups = np.repeat([f"s{i}" for i in range(6)], 20)
    model = ClassicalModel(kind, **settings).fit(x, y, groups)
    p, _ = model.predict(x)
    assert ((p > 0.5) == y).mean() > 0.9
    if kind == "svm":
        assert all(not set(f["fit_sequences"]) & set(f["heldout_sequences"]) for f in model.fold_support)
    model.save(tmp_path / "model.joblib")
    np.testing.assert_array_equal(p, ClassicalModel.load(tmp_path / "model.joblib").predict(x)[0])


def test_prescribed_loss_selection_requires_both_models_and_finite_losses():
    from scripts.successor.select_fusion_models import choose_loss
    results = {kind: {"cross_entropy": {"reference_edtc": 0.8, "brier": 0.02},
                      "squared_error": {"reference_edtc": 0.795, "brier": 0.01}} for kind in ("mlp", "vqc")}
    assert choose_loss(results) == "squared_error"
    results["vqc"]["squared_error"]["brier"] = 0.03
    assert choose_loss(results) == "cross_entropy"
    results["vqc"]["squared_error"] = {"nonfinite": True}
    assert choose_loss(results) == "cross_entropy"
    results["mlp"]["cross_entropy"] = {"nonfinite": True}
    with pytest.raises(FloatingPointError, match="no common"):
        choose_loss(results)
    results["vqc"]["squared_error"] = {"reference_edtc": 0.8, "brier": 0.01}
    assert choose_loss(results) == "squared_error"
