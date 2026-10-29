"""
tracker_naive_fusion.py — Baseline: Naive Late Fusion
=====================================================
EDTCNaiveFusionTracker — dual-stream tracker that replaces Dirichlet belief
fusion (DBF/ABF) with simple probability averaging.

Fusion rule (detection and tracking branches)
---------------------------------------------
    p_fused = 0.5 * b_rgb + 0.5 * b_ir     # element-wise average (K-dim)
    pseudo_u = 1.0 - p_fused[0]            # 1 - p_UAV  (class 0 = UAV)

pseudo_u is used in place of u_fused for all switching conditions.
Everything else — DMS, state machine, SiamCAR init, YOLO inference — is
identical to EDTCFusionTracker.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.edtc_fusion.tracker import (
    EDTCFusionTracker,
    FusionConfig,
)


@dataclass
class NaiveFusionConfig(FusionConfig):
    """FusionConfig without the fusion-mode knobs (unused for naive fusion)."""

    # Inherited fields from FusionConfig are all valid — fusion and lam are
    # carried along but never read inside _fuse().  Callers may ignore them.


class EDTCNaiveFusionTracker(EDTCFusionTracker):
    """Naive Late Fusion: average softmax-like probabilities from both streams.

    The only change relative to EDTCFusionTracker is _fuse(): instead of
    combining opinions via ABF or DBF (subjective logic), we average the
    belief-mass vectors produced by the evidential heads, and use
    ``1 - p_UAV`` as a pseudo-uncertainty signal for state switching.

    All other components (YOLO inference, SiamCAR tracking, DMS, state
    machine) are inherited unchanged.
    """

    def _fuse(self, b_rgb: torch.Tensor, u_rgb, b_ir: torch.Tensor, u_ir):
        """Average fusion: return mean belief and pseudo-uncertainty.

        Parameters
        ----------
        b_rgb, b_ir : Tensor(K,)  — belief-mass vectors from each stream
        u_rgb, u_ir : ignored     — vacuities not used by naive fusion

        Returns
        -------
        b_avg    : Tensor(K,)  — element-wise average belief
        pseudo_u : Tensor()    — 1 - b_avg[0]  (1 - p_UAV)
        """
        b_avg = 0.5 * (b_rgb + b_ir)
        pseudo_u = 1.0 - b_avg[0]
        return b_avg, pseudo_u
