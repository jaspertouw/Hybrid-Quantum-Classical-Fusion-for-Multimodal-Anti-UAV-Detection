"""
belief_fusion.py — Phase 5.1–5.2
=================================
Shape-agnostic ABF and DBF belief fusion for the dual-stream UAV detector.

Used by both:
  • Detection branch  (YOLO evidential scalars from _extract_best_anchor_unc)
  • Tracking branch   (SiamCAR spatial maps from model.track → evi_raw)

All functions operate on arbitrary matching tensor shapes.
Class convention (both branches): index 0 = UAV, index 1 = background, K = 2.

Reference
---------
Jøsang, A. (2016). Subjective Logic. Springer.
Sensoy, M. et al. (2018). Evidential Deep Learning to Quantify Classification Uncertainty. NeurIPS.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "compute_beliefs",
    "abf",
    "compute_conflict",
    "discount",
    "dbf",
]

_EPS = 1e-8  # denominator guard


# ---------------------------------------------------------------------------
# Belief mass extraction
# ---------------------------------------------------------------------------

def compute_beliefs(evi_raw: Tensor, K: int = 2) -> tuple[Tensor, Tensor]:
    """Convert raw evidence logits (SiamCAR cls_evi output) to belief masses
    and vacuity.

    Parameters
    ----------
    evi_raw : Tensor  shape (..., K, ...)
        Raw logits from the evidential head.  For SiamCAR the layout is
        (B, K, H, W); the class dimension is dim=1.
    K : int
        Number of evidential classes (default 2: UAV / background).

    Returns
    -------
    b : Tensor  same shape as evi_raw
        Belief masses b_k = e_k / S,  b_k ∈ [0, 1).
    u : Tensor  evi_raw with the class dimension (dim=1) removed
        Vacuity mass u = K / S,  u ∈ (0, 1].

    Notes
    -----
    Partition of unity: b.sum(dim=1) + u = 1 (exactly, by construction).
    """
    evidence = F.relu(evi_raw)                          # e_k >= 0
    alpha = evidence + 1.0                              # α_k >= 1
    S = alpha.sum(dim=1, keepdim=True)                  # (B, 1, H, W)
    b = evidence / S                                    # (B, K, H, W)
    u = (K / S).squeeze(1)                              # (B, H, W)
    return b, u


# ---------------------------------------------------------------------------
# Averaging Belief Fusion  (5.1 — η = 1, no discounting)
# ---------------------------------------------------------------------------

def abf(
    b_rgb: Tensor,
    u_rgb: Tensor,
    b_th: Tensor,
    u_th: Tensor,
) -> tuple[Tensor, Tensor]:
    """Averaging Belief Fusion (subjective logic consensus operator, η = 1).

    Fuses two Dirichlet opinions (belief vector + vacuity) into one.
    This is the baseline fusion and also the final aggregation step inside DBF.

    Parameters
    ----------
    b_rgb, b_th : Tensor  shape (..., K)  or  (K,)  or  scalar-like
        Belief masses for each class from the RGB and thermal streams.
        Last dimension is the class dimension.
    u_rgb, u_th : Tensor  shape (...)  or  scalar
        Vacuity masses (no class dimension).

    Returns
    -------
    b_fused : Tensor  same shape as b_rgb
    u_fused : Tensor  same shape as u_rgb

    Formula
    -------
    b_fused(c) = [b_rgb(c) · u_th  +  b_th(c) · u_rgb] / [u_rgb + u_th]
    u_fused    = [2 · u_rgb · u_th]                     / [u_rgb + u_th]

    Limiting cases
    --------------
    u_rgb = 1  (vacuous RGB opinion)  =>  b_fused ~ b_th,  u_fused ~ u_th
    u_th  = 1  (vacuous IR opinion)   =>  b_fused ~ b_rgb, u_fused ~ u_rgb
    u_rgb = u_th = 0 (both dogmatic)  =>  b_fused = (b_rgb + b_th) / 2, u_fused = 0

    Note: the identity u_rgb=1 => b_fused=b_th holds exactly only when u_th=0.
    For non-zero u_th the formula gives b_th/(1+u_th), which is the correct
    weighted subjective-logic result.
    """
    u_rgb = torch.as_tensor(u_rgb).float()
    u_th  = torch.as_tensor(u_th).float()

    denom = u_rgb + u_th          # (...), no class dim
    near_zero = denom < _EPS      # boolean mask for degenerate (both dogmatic) case

    # Normal fusion (safe denominator)
    safe_denom   = denom.clamp(min=_EPS)
    safe_denom_b = safe_denom.unsqueeze(-1)   # (..., 1) -> broadcasts over K
    b_norm = (b_rgb * u_th.unsqueeze(-1) + b_th * u_rgb.unsqueeze(-1)) / safe_denom_b
    u_norm = (2.0 * u_rgb * u_th) / safe_denom

    # Degenerate fallback: average beliefs, zero vacuity
    # Partition of unity holds because when u=0, b.sum(-1)=1, so avg.sum(-1)=1.
    b_deg = (b_rgb + b_th) / 2.0
    u_deg = torch.zeros_like(u_norm)

    b_fused = torch.where(near_zero.unsqueeze(-1), b_deg, b_norm)
    u_fused = torch.where(near_zero, u_deg, u_norm)
    return b_fused, u_fused


# ---------------------------------------------------------------------------
# Conflict score
# ---------------------------------------------------------------------------

def compute_conflict(b_rgb: Tensor, b_th: Tensor) -> Tensor:
    """Compute the pairwise conflict score C between two belief vectors.

    C = Σ_{j ≠ k}  b_rgb[j] · b_th[k]

    For K = 2 (UAV / background):
        C = b_rgb[..., 0] · b_th[..., 1]  +  b_rgb[..., 1] · b_th[..., 0]

    For general K (numerically equivalent, no loops):
        C = b_rgb.sum(-1) · b_th.sum(-1)  −  (b_rgb · b_th).sum(-1)
          = (1 − u_rgb) · (1 − u_th)     −  (b_rgb · b_th).sum(-1)
        (the second form avoids redundant sums when u is available)

    Parameters
    ----------
    b_rgb, b_th : Tensor  shape (..., K)
        Belief mass vectors (class dimension last).

    Returns
    -------
    C : Tensor  shape (...)   — class dimension removed.
        C ∈ [0, 1].  C = 0 means no conflict; C = 1 means total conflict.
    """
    # General closed-form: sum of products minus dot product
    C = b_rgb.sum(dim=-1) * b_th.sum(dim=-1) - (b_rgb * b_th).sum(dim=-1)
    return C.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Discounting operator
# ---------------------------------------------------------------------------

def discount(
    b: Tensor,
    u: Tensor,
    C: Tensor,
    lam: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Apply conflict-based discounting to one Dirichlet opinion.

    The discount factor η ∈ [0, 1] attenuates the belief masses and
    re-absorbs the residual mass into vacuity.

    Parameters
    ----------
    b : Tensor  shape (..., K)
        Original belief masses.
    u : Tensor  shape (...)
        Original vacuity.
    C : Tensor  shape (...)
        Conflict score from compute_conflict (same shape as u).
    lam : float
        Discount exponent λ > 0.  λ = 1 gives η = 1 − C (linear; default).
        Larger λ suppresses discounting (η → 1 for all C < 1).

    Returns
    -------
    b_tilde : Tensor  shape (..., K)
        Discounted belief masses:  b̃(c) = η · b(c).
    u_tilde : Tensor  shape (...)
        Updated vacuity:           ũ = 1 − Σ_c b̃(c) = u + C·(1 − u)  [λ=1].

    Formula
    -------
    η = (1 − C^λ)^(1/λ)
    b̃(c) = η · b(c)
    ũ    = 1 − Σ_c b̃(c)

    Limiting behaviour (λ = 1)
    --------------------------
    C → 0  ⟹  η → 1  (no discounting; opinion unchanged)
    C → 1  ⟹  η → 0  (total conflict; all belief absorbed into vacuity)
              ⟹  b̃ → 0,  ũ → 1
    """
    # η = (1 − C^λ)^(1/λ)
    eta = (1.0 - C.clamp(0.0, 1.0) ** lam) ** (1.0 / lam)   # shape (...)

    # Broadcast η to class dimension for b
    eta_b = eta.unsqueeze(-1) if isinstance(eta, Tensor) and eta.dim() > 0 else eta

    b_tilde = eta_b * b                                       # (..., K)
    u_tilde = 1.0 - b_tilde.sum(dim=-1)                      # (...)
    u_tilde = u_tilde.clamp(0.0, 1.0)                        # numerical safety
    return b_tilde, u_tilde


# ---------------------------------------------------------------------------
# Discounted Belief Fusion  (5.2)
# ---------------------------------------------------------------------------

def dbf(
    b_rgb: Tensor,
    u_rgb: Tensor,
    b_th: Tensor,
    u_th: Tensor,
    lam: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Discounted Belief Fusion (DBF).

    Extends ABF with a conflict-aware discounting pre-step that attenuates
    both opinions before aggregation.  When C = 0, DBF reduces to ABF exactly.

    Parameters
    ----------
    b_rgb, b_th : Tensor  shape (..., K)
        Belief masses from RGB and thermal (IR) streams.
    u_rgb, u_th : Tensor  shape (...)
        Vacuity from each stream.
    lam : float
        Discount exponent λ (default 1.0 for linear discounting).

    Returns
    -------
    b_fused : Tensor  shape (..., K)
    u_fused : Tensor  shape (...)

    Pipeline
    --------
    1.  C              = compute_conflict(b_rgb, b_th)
    2.  b̃_rgb, ũ_rgb  = discount(b_rgb, u_rgb, C, lam)
    3.  b̃_th,  ũ_th   = discount(b_th,  u_th,  C, lam)
    4.  b_fused, u_fused = abf(b̃_rgb, ũ_rgb, b̃_th, ũ_th)

    Limiting behaviour (λ = 1)
    --------------------------
    C → 0  ⟹  η → 1  ⟹  DBF ≡ ABF  (no discounting)
    C → 1  ⟹  η → 0  ⟹  ũ_rgb → 1, ũ_th → 1
              ⟹  u_fused → 1  (maximal uncertainty under total conflict)
    """
    C = compute_conflict(b_rgb, b_th)
    b_tilde_rgb, u_tilde_rgb = discount(b_rgb, u_rgb, C, lam)
    b_tilde_th,  u_tilde_th  = discount(b_th,  u_th,  C, lam)
    return abf(b_tilde_rgb, u_tilde_rgb, b_tilde_th, u_tilde_th)
