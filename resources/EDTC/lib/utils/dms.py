"""
dms.py — Phase 5.3
==================
Dynamic Modality Selection for the dual-stream UAV detector.

Selects the bounding-box output from the modality whose per-coordinate
Gaussian variances are smallest after normalizing by predicted box dimensions
(scale invariance).  The returned sigma2_best scalar feeds the downstream
EDTC switching condition.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


def dms(
    box_rgb: Tensor,
    var_rgb: Tensor,
    box_th: Tensor,
    var_th: Tensor,
) -> tuple[Tensor, Tensor]:
    """Dynamic Modality Selection based on normalized spatial uncertainty.

    Parameters
    ----------
    box_rgb, box_th : Tensor  shape (..., 4)
        Predicted bounding boxes [cx, cy, w, h] from RGB and thermal streams.
    var_rgb, var_th : Tensor  shape (..., 4)
        Per-coordinate Gaussian variances [σ²_cx, σ²_cy, σ²_w, σ²_h].

    Returns
    -------
    best_box : Tensor  shape (..., 4)
        Bounding box from the lower-uncertainty modality.
    sigma2_best : Tensor  shape (...)
        Scalar normalized variance of the selected modality.

    Algorithm
    ---------
    1. Normalize each variance by the relevant box dimension:
         σ̂²_cx = σ²_cx / w,  σ̂²_cy = σ²_cy / h,
         σ̂²_w  = σ²_w  / w,  σ̂²_h  = σ²_h  / h
    2. Average the four normalized variances to a scalar σ̂²_m per modality.
    3. Select the modality with the lower σ̂²_m (RGB wins ties).
    """
    # Box dimensions for normalization — clamp prevents division by zero
    w_rgb = box_rgb[..., 2:3].clamp(min=_EPS)
    h_rgb = box_rgb[..., 3:4].clamp(min=_EPS)
    w_th  = box_th[..., 2:3].clamp(min=_EPS)
    h_th  = box_th[..., 3:4].clamp(min=_EPS)

    # Divisors: [w, h, w, h] aligned with [σ²_cx, σ²_cy, σ²_w, σ²_h]
    scale_rgb = torch.cat([w_rgb, h_rgb, w_rgb, h_rgb], dim=-1)
    scale_th  = torch.cat([w_th,  h_th,  w_th,  h_th],  dim=-1)

    sigma2_rgb = (var_rgb / scale_rgb).mean(dim=-1)   # (...)
    sigma2_th  = (var_th  / scale_th ).mean(dim=-1)   # (...)

    use_rgb = sigma2_rgb <= sigma2_th                  # RGB wins ties

    best_box    = torch.where(use_rgb.unsqueeze(-1), box_rgb, box_th)
    sigma2_best = torch.where(use_rgb, sigma2_rgb, sigma2_th)

    return best_box, sigma2_best
