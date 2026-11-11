import unittest

import numpy as np

from src.thesis_experiments import presence_opinions as opinions

try:
    import torch
except ImportError:
    torch = None


class OpinionContractTests(unittest.TestCase):
    def test_evidence_partition_and_vacuous_opinion(self):
        result = opinions.opinion([[0, 0], [10, 1]])
        np.testing.assert_allclose(result["beliefs"].sum(1) + result["uncertainty"], 1)
        np.testing.assert_equal(result["probabilities"][0], [0.5, 0.5])
        with self.assertRaises(ValueError):
            opinions.opinion([[float("nan"), 0]])

    def test_positive_conflict_does_not_imply_opposition(self):
        same = opinions.paired_report([[5, 5]], [[5, 5]])
        self.assertGreater(same["inherited_conflict_mean"], 0)
        self.assertEqual(same["degree_of_conflict_mean"], 0)
        self.assertEqual(same["opposite_decisions"], 0)
        opposite = opinions.paired_report([[20, 0]], [[0, 20]])
        self.assertEqual(opposite["confident_opposite_decisions"], 1)
        self.assertGreater(opposite["degree_of_conflict_mean"], 0.5)
        missing = opinions.paired_report([[20, 0]], [[0, 0]])
        self.assertEqual(missing["degree_of_conflict_mean"], 0)
        self.assertEqual(missing["inherited_conflict_mean"], 0)

    def test_sampling_is_prediction_independent_and_blocks_test(self):
        rows = [{"key": str(i), "split": "train", "rgb": {"present": i < 10}}
                for i in range(20)]
        chosen = opinions.sample_by_class(rows, "rgb", 3, split="train")
        self.assertEqual(len(chosen), 6)
        self.assertEqual(sum(row["rgb"]["present"] for row in chosen), 3)
        self.assertEqual(chosen, opinions.sample_by_class(list(reversed(rows)), "rgb", 3, split="train"))
        with self.assertRaises(ValueError):
            opinions.sample_by_class(rows, "rgb", 3, split="val")
        with self.assertRaises(ValueError):
            opinions.sample_by_class(rows, "rgb", 3, split="test")

    def test_train_only_statistics_and_constant_features(self):
        mean, scale = opinions.fit_standardization([[2, 1], [4, 1]], split="train")
        np.testing.assert_allclose(mean, [3, 1])
        self.assertTrue((scale > 0).all())
        with self.assertRaises(ValueError):
            opinions.fit_standardization([[2], [4]], split="val")

    def test_both_classes_required_and_constant_predictor_fails_gate(self):
        good = opinions.classification_report([[20, 0], [0, 20]], [True, False], population_prevalence=0.8)
        self.assertTrue(good["both_class_learning_gate"])
        bad = opinions.classification_report([[20, 0], [20, 0]], [True, False], population_prevalence=0.8)
        self.assertFalse(bad["both_class_learning_gate"])
        self.assertEqual(bad["background_recall"], 0)
        with self.assertRaises(ValueError):
            opinions.classification_report([[20, 0]], [True], population_prevalence=0.8)


@unittest.skipIf(torch is None, "Torch tests also run before the Snellius pilot")
class OpinionTorchTests(unittest.TestCase):
    def test_loss_gradient_finite_difference_and_prior_weighting(self):
        evidence = torch.tensor([[2.0, 0.4], [0.3, 3.0]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([True, False])
        loss = opinions.evidential_loss(evidence, labels, epoch=3, population_prevalence=0.8)
        loss.backward()
        epsilon = 1e-5
        for i in range(2):
            for j in range(2):
                plus, minus = evidence.detach().clone(), evidence.detach().clone()
                plus[i, j] += epsilon
                minus[i, j] -= epsilon
                numerical = (opinions.evidential_loss(plus, labels, epoch=3, population_prevalence=0.8)
                             - opinions.evidential_loss(minus, labels, epoch=3, population_prevalence=0.8)) / (2 * epsilon)
                self.assertAlmostEqual(evidence.grad[i, j].item(), numerical.item(), places=6)
        # Duplicating one sampled class cannot change the population-weighted loss.
        repeated = evidence.detach()[[0, 0, 1]]
        self.assertAlmostEqual(loss.item(), opinions.evidential_loss(
            repeated, torch.tensor([True, True, False]), epoch=3, population_prevalence=0.8).item())

    def test_head_learns_both_classes_and_roundtrips(self):
        torch.manual_seed(11)
        values = torch.tensor([[-2., 1.], [-1., 1.], [1., 1.], [2., 1.]])
        labels = torch.tensor([False, False, True, True])
        head = opinions.build_head(2)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.03)
        for epoch in range(1, 101):
            optimizer.zero_grad()
            loss = opinions.evidential_loss(head(values), labels, epoch=epoch, population_prevalence=0.5)
            loss.backward()
            optimizer.step()
        result = head(values).detach().numpy()
        self.assertTrue(opinions.classification_report(result, labels.numpy(),
                        population_prevalence=0.5)["both_class_learning_gate"])
        copy = opinions.build_head(2)
        copy.load_state_dict(head.state_dict())
        self.assertTrue(torch.equal(head(values), copy(values)))

    def test_hook_pools_all_maps_without_updating_detector(self):
        class Detect(torch.nn.Module):
            def forward(self, maps):
                return maps[0].sum()

        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(2)
                self.model = torch.nn.ModuleList([Detect()])

            def forward(self, images):
                features = self.bn(images)
                return self.model[-1]([features, features * 2])

        detector = Detector()
        adapter = opinions.FrozenFrameFeatures(detector)
        snapshot = {key: value.clone() for key, value in detector.state_dict().items()}
        images = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
        result = adapter(images)
        with torch.no_grad():
            feature = detector.bn(images)
        expected = torch.cat([feature.mean((-2, -1)), feature.amax((-2, -1)),
                              (feature * 2).mean((-2, -1)), (feature * 2).amax((-2, -1))], 1)
        self.assertTrue(torch.equal(result, expected))
        self.assertFalse(result.requires_grad)
        for key, value in detector.state_dict().items():
            self.assertTrue(torch.equal(value, snapshot[key]))
        adapter.close()
        self.assertEqual(len(detector.model[-1]._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
