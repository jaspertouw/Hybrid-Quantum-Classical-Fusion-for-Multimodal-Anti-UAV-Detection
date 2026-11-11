"""Torch edge cases; runnable with stdlib unittest in the detector environment."""

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    root = Path(__file__).resolve().parents[1] / "resources/EDTC"
    sys.path[:0] = [str(root), str(root / "yolov5")]
    from lib.models.yolo_evidential.postprocess import select_successor_candidate, select_successor_batch


def _scale(beliefs, uncertainty, variance):
    shape = (1, 1, 1, len(beliefs), -1)
    grid = torch.tensor(variance, dtype=torch.float32).reshape(shape)
    return {
        "b": torch.tensor(beliefs, dtype=torch.float32).reshape(shape),
        "u": torch.tensor(uncertainty, dtype=torch.float32).reshape(1, 1, 1, -1, 1),
        "sigma2_grid": grid,
        "sigma2_pixel": grid * 64.0,
        "sigma2_norm_corrected": grid / 100.0,
        "sigma2_norm_inherited": grid / 10.0,
    }


@unittest.skipIf(torch is None, "Torch is available in the Snellius detector environment")
class EvidentialPostprocessTests(unittest.TestCase):
    def test_inclusive_boundary_and_anchor_identity(self):
        prediction = torch.tensor([
            [10, 10, 4, 6, 0.5, 0.5],
            [20, 20, 8, 8, 0.9, 0.2],
        ], dtype=torch.float32)
        scale = _scale([[0.5, 0.25], [0.2, 0.3]], [0.25, 0.5],
                       [[1, 2, 3, 4], [5, 6, 7, 8]])
        selected = select_successor_candidate(prediction, [scale])
        self.assertEqual(selected["anchor_index"], 0)
        self.assertAlmostEqual(selected["candidate_score"].item(), 0.25)
        self.assertAlmostEqual(selected["objectness"].item(), 0.5)
        self.assertEqual(selected["sigma2_grid"].tolist(), [1, 2, 3, 4])

    def test_lowest_anchor_index_wins_exact_tie(self):
        prediction = torch.tensor([
            [10, 10, 4, 6, 0.8, 0.5],
            [10, 10, 4, 6, 0.5, 0.8],
        ], dtype=torch.float32)
        scale = _scale([[0.5, 0.25], [0.8, 0.1]], [0.25, 0.1],
                       [[1, 1, 1, 1], [2, 2, 2, 2]])
        self.assertEqual(select_successor_candidate(prediction, [scale])["anchor_index"], 0)

    def test_nonfinite_uncertainty_is_rejected(self):
        prediction = torch.tensor([[10, 10, 4, 6, 0.8, 0.5]], dtype=torch.float32)
        scale = _scale([[0.5, 0.25]], [0.25], [[1, 1, float("nan"), 1]])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            select_successor_candidate(prediction, [scale])

    def test_batch_preserves_missing_frame_and_scale_offset(self):
        prediction = torch.tensor([
            [[10, 10, 4, 4, 0.1, 0.5], [20, 20, 4, 4, 0.1, 0.6]],
            [[10, 10, 4, 4, 0.1, 0.5], [20, 20, 4, 4, 0.9, 0.6]],
        ], dtype=torch.float32)
        first = _scale([[0.5, 0.25]], [0.25], [[1, 1, 1, 1]])
        second = _scale([[0.6, 0.1]], [0.3], [[2, 2, 2, 2]])
        scales = [{key: value.repeat(2, 1, 1, 1, 1) for key, value in scale.items()}
                  for scale in (first, second)]
        selected = select_successor_batch(prediction, scales)
        self.assertIsNone(selected[0])
        self.assertEqual(selected[1]["anchor_index"], 1)
        self.assertEqual(selected[1]["sigma2_grid"].tolist(), [2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
