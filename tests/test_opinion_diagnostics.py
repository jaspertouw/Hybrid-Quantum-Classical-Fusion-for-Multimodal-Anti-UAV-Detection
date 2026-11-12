import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.thesis_experiments import opinion_diagnostics as diagnostic
from src.thesis_experiments import presence_opinions as opinions

try:
    import torch
except ImportError:
    torch = None


def frames():
    return [{"key": str(i), "split": "train", "sequence": f"s{i % 3}",
             "rgb": {"present": i < 6, "image": f"rgb/{i}.jpg"}} for i in range(12)]


class DiagnosticContractTests(unittest.TestCase):
    def test_samples_are_nested_and_keep_labels(self):
        rows = frames()
        small = opinions.sample_by_class(rows, "rgb", 2, split="train")
        expanded = opinions.sample_by_class(rows, "rgb", 100, split="train")
        self.assertEqual(len(expanded), 12)
        self.assertTrue({row["key"] for row in small} <= {row["key"] for row in expanded})
        report = diagnostic.support(expanded, "rgb")
        self.assertEqual(report["background"]["sequences"], 3)
        self.assertAlmostEqual(report["background"]["largest_sequence_fraction"], 1 / 3)

    def test_cache_reuses_verified_features_and_rejects_tampering(self):
        rows = frames()
        context = diagnostic.feature_context(rows, "rgb", {"detector": "abc"})
        called = []
        def extract():
            called.append(True)
            return np.arange(24, dtype=np.float32).reshape(12, 2)
        with tempfile.TemporaryDirectory() as directory:
            values, record = diagnostic.cached_features(directory, context, extract)
            repeated, _ = diagnostic.cached_features(directory, context, extract)
            np.testing.assert_array_equal(values, repeated)
            self.assertEqual(len(called), 1)
            selected = diagnostic.subset_features(values, rows, [rows[7], rows[1]])
            np.testing.assert_array_equal(selected, values[[7, 1]])
            (Path(record["path"]) / "features.npz").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                diagnostic.cached_features(directory, context, extract)

    def test_cache_blocks_test_and_incomplete_outputs(self):
        rows = frames()
        rows[0]["split"] = "test"
        with self.assertRaises(ValueError):
            diagnostic.feature_context(rows, "rgb", {})
        with tempfile.TemporaryDirectory() as directory:
            context = diagnostic.feature_context(frames(), "rgb", {})
            with self.assertRaises(ValueError):
                diagnostic.cached_features(directory, context, lambda: np.ones((2, 3)))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_sequence_error_counts(self):
        rows = frames()
        evidence = [[10, 0] if row["rgb"]["present"] else [0, 10] for row in rows]
        result = diagnostic.per_sequence_report(evidence, rows, "rgb")
        self.assertEqual(sum(item["background_correct"] for item in result.values()), 6)
        self.assertEqual(sum(item["uav_correct"] for item in result.values()), 6)

    def test_diagnostic_inherits_original_pilot_without_changes(self):
        from src.thesis_experiments.provenance import sha256_file
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root / "experiments/protocol/presence_diagnostic_v1.json").read_text())
        self.assertEqual(spec["pilot_specification_sha256"], sha256_file(root / "experiments/protocol/presence_opinions_v1.json"))
        self.assertEqual(spec["expanded_per_class"], 2048)


@unittest.skipIf(torch is None, "Torch available on Snellius")
class DiagnosticTorchTests(unittest.TestCase):
    def test_class_losses_reconstruct_unchanged_weighted_loss(self):
        evidence = torch.tensor([[2., 1.], [3., 1.], [1., 4.]])
        labels = torch.tensor([True, True, False])
        report = diagnostic.class_losses(evidence, labels, epoch=3)
        expected = opinions.evidential_loss(evidence, labels, epoch=3, population_prevalence=0.9)
        self.assertAlmostEqual(0.9 * report["uav"]["total"] + 0.1 * report["background"]["total"], expected.item(), places=6)

    def test_controlled_training_uses_same_seed_and_no_validation_fit(self):
        rows = frames()
        values = np.array([[float(row["rgb"]["present"]), 1.] for row in rows], dtype=np.float32)
        spec = {"seed": 11, "hidden_size": 16, "learning_rate": 0.001, "epochs": 2}
        report, checkpoint, evidence = diagnostic.train_arm(values, rows, values + 10,
                    rows, "rgb", spec, 0.5, 0.5)
        np.testing.assert_allclose(checkpoint["mean"], values.mean(0))
        self.assertEqual(len(report["training_class_history"]), 1)
        _, repeated, prediction = diagnostic.train_arm(values, rows, values + 10,
                    rows, "rgb", spec, 0.5, 0.5)
        for key in checkpoint["state_dict"]:
            self.assertTrue(torch.equal(checkpoint["state_dict"][key], repeated["state_dict"][key]))
        np.testing.assert_array_equal(evidence, prediction)


if __name__ == "__main__":
    unittest.main()
