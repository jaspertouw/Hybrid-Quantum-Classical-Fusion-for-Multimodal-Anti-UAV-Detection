"""Successor experiment code for the hybrid fusion thesis.

This package is deliberately separate from :mod:`src.edtc_fusion`, which
preserves the inherited baseline implementation.
"""

from .config import ExperimentConfig, load_experiment_config
from .run import RunLayout

__all__ = ["ExperimentConfig", "RunLayout", "load_experiment_config"]
