import unittest

import numpy as np

from src.thesis_experiments import opinion_calibration as calibration
from src.thesis_experiments import opinion_objectives as study
from src.thesis_experiments.presence_opinions import opinion

try:
    import torch
except ImportError:
    torch = None


def example():
    rows = [{"key": f"{s}/{i}", "sequence": f"s{s}", "split": "train",
             "rgb": {"present": i < 4}} for s in range(12) for i in range(8)]
    population = {label: {"per_sequence": {f"s{s}": number for s in range(12)}}
                  for label, number in (("uav", 12), ("background", 4))}
    _, previous = study.training_sequence_split(population, rows, "rgb")
    return rows, population, previous


class CalibrationContractTests(unittest.TestCase):
    def test_three_way_partition_keeps_original_head_fit_and_is_disjoint(self):
        rows, population, previous = example()
        selected, result = calibration.calibration_partition(population, rows, previous, "rgb")
        sets = {name: {row["sequence"] for row in data} for name, data in selected.items()}
        self.assertEqual(sets["head_fit"], set(previous["fit_sequences"]))
        self.assertFalse(sets["calibration"] & sets["evaluation"])
        self.assertFalse(sets["head_fit"] & (sets["calibration"] | sets["evaluation"]))
        self.assertEqual(sets["calibration"] | sets["evaluation"], set(previous["heldout_train_sequences"]))
        self.assertEqual(result["population_prevalence"]["calibration"], 0.75)
        _, repeated = calibration.calibration_partition(population, list(reversed(rows)), previous, "rgb")
        self.assertEqual(result, repeated)
        rows[0]["split"] = "val"
        with self.assertRaises(ValueError):
            calibration.calibration_partition(population, rows, previous, "rgb")

    def test_scaling_preserves_opinion_identities_and_vacuous_dropout(self):
        evidence = [[0, 0], [10, 1], [2, 20]]
        np.testing.assert_array_equal(calibration.apply_scaling(evidence, [0, 0]), evidence)
        result = calibration.apply_scaling(evidence, [2, -1])
        np.testing.assert_array_equal(result[0], [0, 0])
        values = opinion(result)
        np.testing.assert_allclose(values["beliefs"].sum(1) + values["uncertainty"], 1)
        np.testing.assert_allclose(values["probabilities"], values["beliefs"] + values["uncertainty"][:, None] / 2)
        with self.assertRaises(ValueError):
            calibration.apply_scaling(evidence, [7, 0])
        with self.assertRaises(ValueError):
            calibration.apply_scaling([[-1, 2]], [0, 0])

    def test_gate_rejects_better_brier_with_background_collapse(self):
        raw = {"population_weighted_brier": 0.05, "population_weighted_log_loss": 0.2}
        good = {"uav_recall": 0.95, "background_recall": 0.7, "population_weighted_brier": 0.02,
                "population_weighted_log_loss": 0.1, "constant_fitting_prevalence_brier": 0.03}
        self.assertTrue(calibration.joint_gate(raw, good))
        self.assertFalse(calibration.joint_gate(raw, {**good, "background_recall": 0.1}))
        self.assertFalse(calibration.joint_gate(raw, {**good, "population_weighted_brier": 0.04}))


@unittest.skipIf(torch is None, "Torch available on Snellius")
class CalibrationTorchTests(unittest.TestCase):
    def test_fit_improves_fitting_objective_and_is_reproducible(self):
        evidence = np.array([[2, 1], [3, 1], [1, 3], [1, 2]], dtype=float)
        labels = [True, True, False, False]
        fitted = calibration.fit_scaling(evidence, labels, 0.9)
        repeated = calibration.fit_scaling(evidence, labels, 0.9)
        self.assertLess(fitted["objective_after"], fitted["objective_before"])
        np.testing.assert_array_equal(fitted["log_scales"], repeated["log_scales"])
        self.assertFalse(fitted["uncertainty_calibration_claimed"])
        with self.assertRaises(ValueError):
            calibration.fit_scaling(evidence, [True] * 4, 0.9)

    def test_zero_evidence_is_preserved_without_inventing_information(self):
        fitted = calibration.fit_scaling(np.zeros((4, 2)), [True, True, False, False], 0.9)
        np.testing.assert_array_equal(fitted["log_scales"], [0, 0])
        np.testing.assert_array_equal(calibration.apply_scaling(np.zeros((4, 2)), fitted["log_scales"]), 0)


if __name__ == "__main__":
    unittest.main()
