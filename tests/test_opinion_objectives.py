import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.thesis_experiments import opinion_diagnostics as diagnostic
from src.thesis_experiments import opinion_objectives as study

try:
    import torch
except ImportError:
    torch = None


def example():
    rows = [{"key": f"{sequence}/{i}", "sequence": f"s{sequence}", "split": "train",
             "rgb": {"present": i < 4, "image": f"train/{sequence}/{i}.jpg"}}
            for sequence in range(9) for i in range(8)]
    population = {label: {"per_sequence": {f"s{i}": count for i in range(9)}}
                  for label, count in (("uav", 12), ("background", 4))}
    return rows, population


class ObjectiveContractTests(unittest.TestCase):
    def test_partition_is_fixed_disjoint_and_uses_population_prior(self):
        rows, population = example()
        selected, split = study.training_sequence_split(population, rows, "rgb")
        self.assertEqual(len(split["heldout_train_sequences"]), 3)
        self.assertFalse(set(split["fit_sequences"]) & set(split["heldout_train_sequences"]))
        self.assertEqual(split["population_prevalence"]["fit"], 0.75)
        other, repeated = study.training_sequence_split(population, list(reversed(rows)), "rgb")
        self.assertEqual(split, repeated)
        self.assertEqual({row["key"] for row in selected["fit"]}, {row["key"] for row in other["fit"]})
        rows[0]["split"] = "val"
        with self.assertRaises(ValueError):
            study.training_sequence_split(population, rows, "rgb")

    def test_no_split_search_when_support_is_missing(self):
        rows, population = example()
        for row in rows:
            row["rgb"]["present"] = True
        with self.assertRaisesRegex(ValueError, "no automatic split search"):
            study.training_sequence_split(population, rows, "rgb")

    def test_training_cache_is_reused_and_validation_is_blocked_before_loading(self):
        rows, _ = example()
        with tempfile.TemporaryDirectory() as directory:
            context = diagnostic.feature_context(rows, "rgb", {"phase2_sha256": "abc", "source_sha256": {}})
            values, record = diagnostic.cached_features(directory, context, lambda: np.ones((len(rows), 4)))
            read, membership = study.read_training_features(record, "rgb", "abc", Path(directory))
            np.testing.assert_equal(values, read)
            self.assertEqual([row["key"] for row in rows], [row["key"] for row in membership])
            marker = Path(record["path"]) / "COMPLETED.json"
            metadata = json.loads(marker.read_text())
            metadata["context"]["split"] = "val"
            marker.write_text(json.dumps(metadata))
            with patch.object(np, "load", side_effect=AssertionError("must not load validation")):
                with self.assertRaises(ValueError):
                    study.read_training_features(record, "rgb", "abc", Path(directory))

    def test_probability_quality_uses_natural_prior_not_balanced_prior(self):
        rows = [{"rgb": {"present": True}}, {"rgb": {"present": False}}]
        report = study.probability_report([[8, 0], [3, 0]], rows, "rgb", 0.75, 0.6)
        self.assertAlmostEqual(report["population_weighted_brier"], 0.75 * 0.01 + 0.25 * 0.64)
        self.assertAlmostEqual(report["constant_fitting_prevalence_brier"], 0.75 * 0.16 + 0.25 * 0.36)


@unittest.skipIf(torch is None, "Torch available on Snellius")
class ObjectiveTorchTests(unittest.TestCase):
    def test_controlled_objectives_and_train_only_statistics(self):
        rows, population = example()
        selected, split = study.training_sequence_split(population, rows, "rgb")
        features = np.array([[float(row["rgb"]["present"]), float(i % 3)] for i, row in enumerate(rows)], dtype=np.float32)
        spec = {"seed": 11, "hidden_size": 16, "epochs": 2, "learning_rate": 0.001}
        natural, _, = study.fit_objective(features, rows, selected, split, "rgb", spec, "natural_prevalence")
        balanced, checkpoint = study.fit_objective(features, rows, selected, split, "rgb", spec, "class_balanced")
        self.assertEqual(natural["objective_uav_weight"], 0.75)
        self.assertEqual(balanced["objective_uav_weight"], 0.5)
        self.assertEqual(balanced["train"]["constant_prevalence_brier"], 0.75 * 0.25)
        changed = features.copy()
        heldout_sequences = set(split["heldout_train_sequences"])
        for i, row in enumerate(rows):
            if row["sequence"] in heldout_sequences:
                changed[i] += 100
        _, repeated = study.fit_objective(changed, rows, selected, split, "rgb", spec, "class_balanced")
        self.assertEqual(checkpoint["mean"], repeated["mean"])
        for key in checkpoint["state_dict"]:
            self.assertTrue(torch.equal(checkpoint["state_dict"][key], repeated["state_dict"][key]))

    def test_sequence_overlap_is_rejected(self):
        rows, population = example()
        selected, split = study.training_sequence_split(population, rows, "rgb")
        selected["heldout_train"].append(selected["fit"][0])
        with self.assertRaisesRegex(ValueError, "sequence leakage"):
            study.fit_objective(np.ones((len(rows), 4)), rows, selected, split, "rgb", {}, "class_balanced")


if __name__ == "__main__":
    unittest.main()
