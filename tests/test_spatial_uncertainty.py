import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path("resources/EDTC/lib/utils/spatial_uncertainty.py").resolve()
SPEC = importlib.util.spec_from_file_location("spatial_uncertainty", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_corrected_variance_is_dimensionless_and_stride_aware():
    sigma2_grid = np.array([[4.0, 9.0, 16.0, 25.0]])
    width = np.array([[40.0]])
    height = np.array([[20.0]])
    result = np.concatenate(
        MODULE.corrected_variance_components(sigma2_grid, width, height, 8.0), axis=-1
    )
    expected = np.array([[
        4.0 * 64.0 / 40.0**2,
        9.0 * 64.0 / 20.0**2,
        16.0 * 64.0 / 40.0**2,
        25.0 * 64.0 / 20.0**2,
    ]])
    np.testing.assert_allclose(result, expected)


def test_corrected_variance_clamps_subpixel_box_dimensions():
    sigma2_grid = np.ones((1, 4))
    width = np.array([[0.25]])
    height = np.array([[0.0]])
    result = np.concatenate(
        MODULE.corrected_variance_components(sigma2_grid, width, height, 2.0), axis=-1
    )
    np.testing.assert_allclose(result, np.full((1, 4), 4.0))


def test_inherited_formula_remains_available_for_audit():
    sigma2_grid = np.ones((1, 4))
    width = np.array([[4.0]])
    height = np.array([[2.0]])
    inherited = np.concatenate(
        MODULE.inherited_variance_components(sigma2_grid, width, height), axis=-1
    )
    assert not np.isclose(inherited[0, 0], inherited[0, 2])
