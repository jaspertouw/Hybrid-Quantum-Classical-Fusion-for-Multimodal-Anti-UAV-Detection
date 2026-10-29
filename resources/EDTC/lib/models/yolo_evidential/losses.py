"""
Loss functions for the evidential + Gaussian bbox detection head (Phase 3.2).

Three loss components:

1. Objectness loss (BCEWithLogitsLoss) — unchanged from YOLOv5 baseline.
2. Gaussian bbox NLL loss — replaces CIoU with negative log-likelihood of a
   factored Gaussian over [cx, cy, w, h] coordinates.  CIoU on the means is
   also included as an auxiliary signal (Choi et al., 2019).
3. Dirichlet evidential loss — replaces BCE class loss with the Type II
   Maximum Likelihood Dirichlet loss of Sensoy et al. (2018), plus a
   KL-divergence regulariser that anneals in over training.

References:
  He et al. (2019) — "Bounding Box Regression with Uncertainty for Accurate
    Object Detection."  CVPR 2019.
  Sensoy et al. (2018) — "Evidential Deep Learning to Quantify Classification
    Uncertainty."  NeurIPS 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov5.utils.metrics import bbox_iou
from yolov5.utils.torch_utils import de_parallel


# ---------------------------------------------------------------------------
# Gaussian bbox NLL loss (He et al. 2019)
# ---------------------------------------------------------------------------

def gaussian_bbox_nll(pred_mean, log_var, target, reduction='mean'):
    """Negative log-likelihood of a diagonal Gaussian over bbox coordinates.

    For each coordinate k:
        L_k = 0.5 * log_var_k + 0.5 * (μ_k - y_k)² * exp(-log_var_k)

    This matches -log N(y | μ, σ²) up to a constant (He et al. 2019).

    Args:
        pred_mean: (N, 4) predicted means [cx, cy, w, h] in grid-normalised
                   space (same coordinate frame as target).
        log_var:   (N, 4) predicted log-variances; exp(log_var) = σ².
        target:    (N, 4) ground-truth coordinates.
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        Scalar loss (if reduction != 'none'), else (N, 4) tensor.
    """
    # Clamp log_var before entering the exponential.
    # Without clamping, two failure modes occur:
    #   log_var → -∞  ⟹  (-∞) + exp(+∞)·sq_err  →  NaN
    #   log_var → -∞  with sq_err ≈ 0  ⟹  loss → -∞  (variance collapse)
    # Range [-6, 6] keeps σ² in [~0.002, ~403], wide enough for any
    # YOLOv5 grid-normalised bbox coordinate.
    log_var = log_var.clamp(-6, 6)

    sq_err = (pred_mean - target) ** 2                        # (N, 4)
    nll    = 0.5 * log_var + 0.5 * sq_err * torch.exp(-log_var)  # (N, 4)

    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    return nll


# ---------------------------------------------------------------------------
# Dirichlet evidential classification loss (Sensoy et al. 2018)
# ---------------------------------------------------------------------------

def dirichlet_ece_loss(evidence, y_onehot, K: int, lam: float = 1.0):
    """Expected Cross-Entropy loss under a Dirichlet distribution.

    L = Σ_k y_k [ψ(S) - ψ(α_k)]  +  λ * KL[Dir(ã) || Dir(1,...,1)]

    where
        α_k = e_k + 1,  S = Σ α_k,  ψ = digamma,
        ã_k = y_k + (1 - y_k) * α_k  (remove true-class evidence for KL).

    Args:
        evidence:  (N, K) ReLU-activated evidence values (e_k ≥ 0).
        y_onehot:  (N, K) one-hot ground-truth class labels.
        K:         number of evidential classes.
        lam:       KL regularisation weight (typically annealed from 0→1).

    Returns:
        Scalar loss.
    """
    alpha = evidence + 1.0                            # Dirichlet parameters
    S     = alpha.sum(dim=-1, keepdim=True)           # precision

    # ── Expected CE ───────────────────────────────────────────────────
    # Σ_k y_k [ψ(S) - ψ(α_k)]
    ece = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)  # (N,)

    # ── KL divergence regulariser ─────────────────────────────────────
    # Remove evidence of the true class so the model is not penalised for
    # being confident about the correct class.
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha   # (N, K)
    S_tilde     = alpha_tilde.sum(-1, keepdim=True)

    kl = (
        torch.lgamma(S_tilde)
        - torch.lgamma(torch.tensor(K, dtype=alpha.dtype, device=alpha.device))
        - torch.lgamma(alpha_tilde).sum(-1, keepdim=True)
        + ((alpha_tilde - 1.0) * (
            torch.digamma(alpha_tilde) - torch.digamma(S_tilde)
        )).sum(-1, keepdim=True)
    ).squeeze(-1)                                         # (N,)

    return (ece + lam * kl).mean()


# ---------------------------------------------------------------------------
# Combined evidential loss (drop-in for ComputeLoss)
# ---------------------------------------------------------------------------

class ComputeEvidentialLoss:
    """Combined loss for EvidentialYOLO training.

    Replaces the standard ComputeLoss with:
      - Unchanged objectness BCE loss.
      - Gaussian NLL bbox loss (replaces CIoU).  CIoU auxiliary included.
      - Dirichlet evidential classification loss.

    Usage::
        loss_fn = ComputeEvidentialLoss(model, hyp, lambda_var=0.1,
                                        lambda_evi=1.0)
        loss, components = loss_fn((base_raw, var_raw, evi_raw), targets)

    Args:
        model:       EvidentialYOLO instance.
        hyp:         YOLOv5 hyperparameter dict (keys: 'box', 'obj', 'cls',
                     'anchor_t', 'fl_gamma', 'cls_pw', 'obj_pw',
                     'label_smoothing').
        lambda_var:  weight on the Gaussian variance NLL loss.
        lambda_evi:  weight on the evidential classification loss.
        K:           number of evidential classes (default 2).
    """

    def __init__(self, model, hyp: dict, lambda_var: float = 0.1,
                 lambda_evi: float = 1.0, K: int = 2):
        self.hyp        = hyp
        self.lambda_var = lambda_var
        self.lambda_evi = lambda_evi
        self.K          = K

        head = de_parallel(model).model[-1]   # EvidentialDetect
        self.na      = head.na
        self.nc      = head.nc
        self.nl      = head.nl
        self.anchors = head.anchors
        self.device  = next(model.parameters()).device

        # Objectness criterion (unchanged)
        self.BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyp.get('obj_pw', 1.0)], device=self.device)
        )

        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25])
        self.gr = 1.0    # IoU ratio used for objectness target

        # KL annealing schedule counter (incremented each call)
        self._step = 0
        self._anneal_steps = 1000  # steps until λ_KL reaches 1.0

    @property
    def _kl_weight(self):
        """Linear annealing of the KL term from 0 → 1."""
        return min(1.0, self._step / max(self._anneal_steps, 1))

    def __call__(self, predictions, targets):
        """
        Args:
            predictions: (base_raw, var_raw, evi_raw) from EvidentialDetect.forward()
              base_raw: list[Tensor(bs, na, H, W, no)]
              var_raw:  list[Tensor(bs, na, H, W, 4)]
              evi_raw:  list[Tensor(bs, na, H, W, K)]
            targets: Tensor(num_gt, 6)  — [img_idx, class, cx, cy, w, h] normalised

        Returns:
            (total_loss * batch_size,
             Tensor([lbox, lobj, levi, lvar]))
        """
        self._step += 1
        base_raw, var_raw, evi_raw = predictions

        lbox = torch.zeros(1, device=self.device)
        lvar = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        levi = torch.zeros(1, device=self.device)

        tcls, tbox, indices, anchors = self._build_targets(base_raw, targets)

        for i, pb in enumerate(base_raw):
            b, a, gj, gi = indices[i]    # img, anchor, grid-y, grid-x
            tobj = torch.zeros(pb.shape[:4], dtype=pb.dtype, device=self.device)

            n = b.shape[0]
            if n:
                # ── Positive-anchor predictions ───────────────────────
                pxy   = pb[b, a, gj, gi, :2]                  # (N,2) raw cx/cy
                pwh   = pb[b, a, gj, gi, 2:4]                 # (N,2) raw w/h
                pobj  = pb[b, a, gj, gi, 4]                   # (N,)  raw obj

                pv    = var_raw[i][b, a, gj, gi]               # (N,4) log-vars
                pe    = evi_raw[i][b, a, gj, gi]               # (N,K) raw evi

                # Activate bbox means (identical to Detect)
                pxy_s = pxy.sigmoid() * 2 - 0.5
                pwh_s = (pwh.sigmoid() * 2) ** 2 * anchors[i] # absolute wh in grid units
                pbox  = torch.cat((pxy_s, pwh_s), 1)          # (N,4) predicted

                # ── CIoU on means (auxiliary) ─────────────────────────
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()

                # ── Gaussian NLL on raw logit space ───────────────────
                # We compute NLL on the pre-activation coordinates so that
                # the scale is consistent with the training regression targets.
                pred_mean_raw = torch.cat([pxy, pwh], dim=1)  # (N,4)
                # Target in raw logit space: invert the sigmoid activation
                # cy_raw = sigmoid^{-1}((ty + 0.5) / 2)  for xy
                # wh_raw = sigmoid^{-1}(sqrt(twh / anchor) / 2)
                # This is complex, so we use grid-normalised pbox and tbox directly.
                lvar += self.lambda_var * gaussian_bbox_nll(pbox, pv, tbox[i])

                # ── Objectness target ─────────────────────────────────
                iou_det = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = (
                    (1.0 - self.gr) + self.gr * iou_det
                )

                # ── Evidential classification loss ────────────────────
                # Build one-hot: class 0 = UAV (all targets are UAV; class index
                # from dataset is 0), class 1 = background.
                evidence = F.relu(pe)                          # (N, K) guaranteed ≥ 0
                y_oh = torch.zeros(n, self.K, device=self.device)
                y_oh[range(n), tcls[i].clamp(0, self.K - 1)] = 1.0
                levi += self.lambda_evi * dirichlet_ece_loss(
                    evidence, y_oh, self.K, lam=self._kl_weight
                )

            # ── Objectness loss (all anchors) ─────────────────────────
            lobj += self.BCEobj(pb[..., 4], tobj) * self.balance[i]

        bs = base_raw[0].shape[0]
        lbox *= self.hyp.get('box', 0.05)
        lobj *= self.hyp.get('obj', 1.0)
        levi *= self.hyp.get('cls', 0.5)
        # lvar already weighted by lambda_var

        total = (lbox + lvar + lobj + levi) * bs
        components = torch.cat([lbox, lobj, levi, lvar]).detach()
        return total, components

    # ------------------------------------------------------------------
    # Target builder (adapted from ComputeLoss.build_targets)
    # ------------------------------------------------------------------

    def _build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        g = 0.5
        off = torch.tensor(
            [[0,0],[1,0],[0,1],[-1,0],[0,-1]],
            device=self.device
        ).float() * g

        for i in range(self.nl):
            anchors_i, shape = self.anchors[i], p[i].shape  # (na,2), (bs,na,H,W,no)
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]

            t = targets * gain
            if nt:
                r = t[..., 4:6] / anchors_i[:, None]
                j = torch.max(r, 1.0 / r).max(2)[0] < self.hyp.get('anchor_t', 4.0)
                t = t[j]
                gxy  = t[:, 2:4]
                gxi  = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat(5, 1, 1)[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            indices.append((
                b, a,
                gj.clamp_(0, shape[2] - 1),
                gi.clamp_(0, shape[3] - 1),
            ))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors_i[a])
            tcls.append(c)

        return tcls, tbox, indices, anch
