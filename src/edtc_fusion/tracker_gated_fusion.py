"""
tracker_gated_fusion.py — Baseline: Gated Fusion
=================================================
EDTCGatedFusionTracker — dual-stream tracker that replaces Dirichlet belief
fusion with a learned scalar gate MLP.

Fusion rule
-----------
    g       = sigmoid( MLP( cat([b_rgb, b_ir]) ) )   # ∈ (0, 1)
    b_gated = g * b_rgb + (1 - g) * b_ir
    pseudo_u = 1 - b_gated[0]                        # 1 - p_UAV

The gate MLP is shared between the detection and tracking phases (same
weights applied on peak-extracted belief vectors in both cases).

Gate architecture:  Linear(2K → 16) → ReLU → Linear(16 → 1) → Sigmoid
                    ~100 trainable parameters, trained by train_gate.py.

Everything else — DMS, state machine, SiamCAR init, YOLO inference — is
inherited unchanged from EDTCFusionTracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.edtc_fusion.tracker import (
    EDTCFusionTracker,
    FusionConfig,
)


# ===========================================================================
# Gate modules
# ===========================================================================

class GateDetectionMLP(nn.Module):
    """Scalar gate for detection (and tracking) belief-vector fusion.

    Input  : concatenated belief vectors [b_rgb; b_ir] — shape (2K,)
    Output : scalar gate g ∈ (0, 1)
    """

    def __init__(self, K: int = 2, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * K, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., 2K)  →  (..., 1) gate scalar."""
        return self.net(x)


class GateTrackingConv(nn.Module):
    """Spatial gate for tracking score-map fusion (used during training).

    Input  : concatenated score maps — shape (B, 2, H, W)
             channel 0 = RGB UAV score, channel 1 = IR UAV score
    Output : per-pixel gate g ∈ (0, 1) — shape (B, 1, H, W)

    At inference time the EDTCGatedFusionTracker applies GateDetectionMLP on
    peak-extracted beliefs instead (equivalent to evaluating this gate at the
    peak location), so GateTrackingConv is used only during training.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 2, H, W)  →  (B, 1, H, W) gate map."""
        return self.net(x)


class GateFusionModule(nn.Module):
    """Container for both gate networks.  Handles save / load.

    Parameters
    ----------
    K      : number of classes (default 2: UAV + background)
    hidden : hidden units in both gate networks (default 16)
    """

    def __init__(self, K: int = 2, hidden: int = 16):
        super().__init__()
        self.detect_gate  = GateDetectionMLP(K=K, hidden=hidden)
        self.tracking_gate = GateTrackingConv(hidden=hidden)

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device, K: int = 2, hidden: int = 16) -> "GateFusionModule":
        obj = cls(K=K, hidden=hidden)
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        obj.load_state_dict(state)
        return obj.to(device).eval()


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class GatedFusionConfig(FusionConfig):
    """FusionConfig extended with the gate checkpoint path."""

    gate_weights: str = ""    # path to GateFusionModule checkpoint (from train_gate.py)
    gate_hidden:  int = 16    # hidden units — must match the trained checkpoint


# ===========================================================================
# Tracker
# ===========================================================================

class EDTCGatedFusionTracker(EDTCFusionTracker):
    """Gated Fusion tracker: learned scalar gate replaces DBF/ABF.

    A small MLP (GateDetectionMLP) takes concatenated belief vectors from
    both streams and produces a gate g ∈ (0, 1).  The fused belief is:

        b_gated = g * b_rgb + (1 - g) * b_ir

    The gate MLP is shared across detection and tracking phases so the same
    ~100 parameters are applied whenever a fusion decision is made.

    The gate weights must be trained via train_gate.py before evaluation.
    """

    def __init__(self, config: GatedFusionConfig):
        super().__init__(config)

        if not config.gate_weights:
            raise ValueError(
                "GatedFusionConfig.gate_weights must point to a trained gate "
                "checkpoint (produced by train_gate.py)."
            )

        self.gate = GateFusionModule.load(
            config.gate_weights,
            device=self.device,
            K=config.K,
            hidden=config.gate_hidden,
        )
        self.gate.eval()

    def _fuse(self, b_rgb: torch.Tensor, u_rgb, b_ir: torch.Tensor, u_ir):
        """Gated fusion: MLP gate on concatenated beliefs.

        Parameters
        ----------
        b_rgb, b_ir : Tensor(K,)  — belief-mass vectors from each stream
        u_rgb, u_ir : ignored     — vacuities not used by gated fusion

        Returns
        -------
        b_gated  : Tensor(K,)  — weighted blend of beliefs
        pseudo_u : Tensor()    — 1 - b_gated[0]  (1 - p_UAV)
        """
        b_rgb_d    = b_rgb.to(self.device)
        b_ir_d     = b_ir.to(self.device)
        gate_input = torch.cat([b_rgb_d, b_ir_d])                              # (2K,)
        with torch.no_grad():
            g = self.gate.detect_gate(gate_input).squeeze()                    # scalar, on device
        b_gated  = (g * b_rgb_d + (1.0 - g) * b_ir_d).cpu()                  # keep beliefs on CPU like _ZERO_UNC
        pseudo_u = 1.0 - b_gated[0]
        return b_gated, pseudo_u
