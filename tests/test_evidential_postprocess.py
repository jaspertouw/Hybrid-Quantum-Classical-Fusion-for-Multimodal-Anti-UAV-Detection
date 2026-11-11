import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path("resources/EDTC").resolve()))

from lib.models.yolo_evidential.postprocess import select_successor_candidate


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


def test_direct_selection_preserves_exact_anchor_and_inclusive_boundary():
    prediction = torch.tensor([
        [10, 10, 4, 6, 0.5, 0.5],
        [20, 20, 8, 8, 0.9, 0.2],
    ], dtype=torch.float32)
    scale = _scale(
        beliefs=[[0.5, 0.25], [0.2, 0.3]],
        uncertainty=[0.25, 0.5],
        variance=[[1, 2, 3, 4], [5, 6, 7, 8]],
    )
    selected = select_successor_candidate(prediction, [scale])
    assert selected["anchor_index"] == 0
    assert selected["candidate_score"].item() == pytest.approx(0.25)
    assert selected["objectness"].item() == pytest.approx(0.5)
    assert selected["sigma2_grid"].tolist() == [1, 2, 3, 4]


def test_direct_selection_uses_lowest_anchor_index_for_exact_tie():
    prediction = torch.tensor([
        [10, 10, 4, 6, 0.8, 0.5],
        [10, 10, 4, 6, 0.5, 0.8],
    ], dtype=torch.float32)
    scale = _scale(
        beliefs=[[0.5, 0.25], [0.8, 0.1]],
        uncertainty=[0.25, 0.1],
        variance=[[1, 1, 1, 1], [2, 2, 2, 2]],
    )
    assert select_successor_candidate(prediction, [scale])["anchor_index"] == 0


def test_direct_selection_rejects_non_finite_uncertainty():
    prediction = torch.tensor([[10, 10, 4, 6, 0.8, 0.5]], dtype=torch.float32)
    scale = _scale([[0.5, 0.25]], [0.25], [[1, 1, float("nan"), 1]])
    with pytest.raises(RuntimeError, match="non-finite"):
        select_successor_candidate(prediction, [scale])
