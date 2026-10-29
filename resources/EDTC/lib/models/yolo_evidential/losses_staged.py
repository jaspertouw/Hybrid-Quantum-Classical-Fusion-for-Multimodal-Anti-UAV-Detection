"""
Staged loss functions for Phase 3.3 revised training protocol.

Two separate loss classes, one per training stage:

1. ComputeGaussianBboxLoss (Stage 1)
   Objectness BCE + CIoU auxiliary + Gaussian NLL on bbox coordinates.
   The evidential head (m_evi) is completely ignored — it receives no gradient.
   This lets the Gaussian head stabilise on clean detection supervision before
   any EDL loss is introduced.

2. ComputeEvidentialOnlyLoss (Stage 2)
   EDL Dirichlet loss only (expected CE + KL regularisation toward uniform prior).
   bbox/objectness branches are frozen; no gradient flows to them.
   KL weight is annealed linearly from 0 → 1 over training.

Both classes share the same _build_targets() anchor-matching logic as
ComputeEvidentialLoss in losses.py (copied here to keep the two loss files
self-contained and independently importable).

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

# Import existing primitives — no duplication
from lib.models.yolo_evidential.losses import gaussian_bbox_nll, dirichlet_ece_loss


# ---------------------------------------------------------------------------
# Stage 1: Gaussian bbox loss  (objectness + CIoU + Gaussian NLL)
# ---------------------------------------------------------------------------

class ComputeGaussianBboxLoss:
    """Detection loss for Stage 1 Gaussian head training.

    Combines:
      - Standard objectness BCE (unchanged from YOLOv5 baseline).
      - CIoU loss on bbox means as an auxiliary anchor signal.
      - Gaussian NLL loss on the log-variance branch (m_var).

    The evidential head (m_evi / evi_raw) is completely ignored:
    even if it is included in predictions, it receives zero gradient.

    Args:
        model:       EvidentialYOLO instance.
        hyp:         YOLOv5 hyperparameter dict.
        lambda_var:  weight multiplier on the Gaussian NLL term.
    """

    def __init__(self, model, hyp: dict, lambda_var: float = 0.1):
        self.hyp        = hyp
        self.lambda_var = lambda_var

        head = de_parallel(model).model[-1]   # EvidentialDetect
        self.na      = head.na
        self.nc      = head.nc
        self.nl      = head.nl
        self.anchors = head.anchors
        self.device  = next(model.parameters()).device

        self.BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyp.get('obj_pw', 1.0)], device=self.device)
        )
        self.balance = {3: [4.0, 1.0, 0.4], 4: [8.0, 4.0, 1.0, 0.4]}.get(
            self.nl, [4.0 / (2 ** max(0, i - 1)) for i in range(self.nl)]
        )
        self.gr = 1.0

    def __call__(self, predictions, targets):
        """
        Args:
            predictions: (base_raw, var_raw, evi_raw)  — evi_raw is ignored.
              base_raw: list[Tensor(bs, na, H, W, no)]  — bbox + obj (+ cls ignored)
              var_raw:  list[Tensor(bs, na, H, W, 4)]   — log-variances
              evi_raw:  list[Tensor(bs, na, H, W, K)]   — IGNORED (no gradient)
            targets: Tensor(num_gt, 6) [img_idx, class, cx, cy, w, h] normalised.

        Returns:
            (total_loss * bs,  Tensor([lbox, lobj, lvar]))
        """
        base_raw, var_raw, _evi_raw = predictions   # evi_raw discarded

        lbox = torch.zeros(1, device=self.device)
        lvar = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)

        tcls, tbox, indices, anchors = self._build_targets(base_raw, targets)

        for i, pb in enumerate(base_raw):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pb.shape[:4], dtype=pb.dtype, device=self.device)

            n = b.shape[0]
            if n:
                pxy  = pb[b, a, gj, gi, :2]   # (N,2) raw cx/cy
                pwh  = pb[b, a, gj, gi, 2:4]  # (N,2) raw w/h
                pobj = pb[b, a, gj, gi, 4]    # (N,)  raw obj (used for tobj below)
                pv   = var_raw[i][b, a, gj, gi]  # (N,4) log-variances

                # ── Activate bbox means ───────────────────────────────
                pxy_s = pxy.sigmoid() * 2 - 0.5
                pwh_s = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox  = torch.cat((pxy_s, pwh_s), 1)          # (N,4)

                # ── CIoU auxiliary ────────────────────────────────────
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()

                # ── Gaussian NLL on bbox ──────────────────────────────
                lvar += self.lambda_var * gaussian_bbox_nll(pbox, pv, tbox[i])

                # ── Objectness target ─────────────────────────────────
                iou_det = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou_det

            lobj += self.BCEobj(pb[..., 4], tobj) * self.balance[i]

        bs = base_raw[0].shape[0]
        lbox *= self.hyp.get('box', 0.05)
        lobj *= self.hyp.get('obj', 1.0)
        # lvar already weighted by lambda_var inside the loop

        total = (lbox + lvar + lobj) * bs
        components = torch.cat([lbox, lobj, lvar]).detach()
        return total, components

    # ------------------------------------------------------------------
    # Target builder (identical logic to ComputeEvidentialLoss)
    # ------------------------------------------------------------------

    def _build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(7, device=self.device)
        ai   = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        g   = 0.5
        off = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
            device=self.device
        ).float() * g

        for i in range(self.nl):
            anchors_i, shape = self.anchors[i], p[i].shape
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
                j    = torch.stack((torch.ones_like(j), j, k, l, m))
                t    = t.repeat(5, 1, 1)[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t       = targets[0]
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


# ---------------------------------------------------------------------------
# Stage 2: Evidential-only loss  (EDL Dirichlet loss, no bbox/obj gradient)
# ---------------------------------------------------------------------------

class ComputeEvidentialOnlyLoss:
    """EDL Dirichlet loss for Stage 2 evidential head training.

    Only the m_evi branch (evi_raw) contributes to this loss.
    bbox (base_raw) and log-variance (var_raw) branches are frozen and
    receive zero gradient.

    Loss = E[CE under Dirichlet]  +  λ_KL * KL[Dir(ã) || Dir(1,...,1)]

    KL weight is annealed linearly from 0 → 1 over _anneal_steps forward
    calls so that the model first fits the expected CE signal before the
    regulariser pulls predictions toward the uniform prior.

    Args:
        model:       EvidentialYOLO instance (m_evi trainable, rest frozen).
        hyp:         YOLOv5 hyperparameter dict (only anchor_t used for target building).
        lambda_evi:  overall weight on the EDL loss term.
        K:           number of evidential classes (default 2: UAV, background).
    """

    def __init__(self, model, hyp: dict, lambda_evi: float = 1.0, K: int = 2):
        self.hyp        = hyp
        self.lambda_evi = lambda_evi
        self.K          = K

        head = de_parallel(model).model[-1]
        self.na      = head.na
        self.nc      = head.nc
        self.nl      = head.nl
        self.anchors = head.anchors
        self.device  = next(model.parameters()).device

        # KL annealing counter
        self._step         = 0
        self._anneal_steps = 1000   # overridden by caller to total_steps

    @property
    def _kl_weight(self):
        """Linear annealing of the KL term from 0 → 1."""
        return min(1.0, self._step / max(self._anneal_steps, 1))

    def __call__(self, predictions, targets):
        """
        Args:
            predictions: (base_raw, var_raw, evi_raw)
              base_raw / var_raw: used only for target building (no gradient needed).
              evi_raw: list[Tensor(bs, na, H, W, K)] — receives gradient.
            targets: Tensor(num_gt, 6) [img_idx, class, cx, cy, w, h] normalised.

        Returns:
            (total_loss * bs,  Tensor([levi]))
        """
        self._step += 1
        base_raw, _var_raw, evi_raw = predictions

        levi = torch.zeros(1, device=self.device)

        # Build targets using base_raw shapes (anchors/grid sizes) —
        # no gradient flows to base_raw since backbone is frozen.
        tcls, tbox, indices, anchors = self._build_targets(base_raw, targets)

        for i in range(self.nl):
            pb = base_raw[i]          # shape reference only — no gradient used
            b, a, gj, gi = indices[i]

            n = b.shape[0]
            if n:
                pe = evi_raw[i][b, a, gj, gi]   # (N, K) — gradient flows here

                # One-hot targets: class 0 = UAV for all positive anchors
                evidence = F.relu(pe)
                y_oh = torch.zeros(n, self.K, device=self.device)
                y_oh[range(n), tcls[i].clamp(0, self.K - 1)] = 1.0

                levi += self.lambda_evi * dirichlet_ece_loss(
                    evidence, y_oh, self.K, lam=self._kl_weight
                )

        bs    = base_raw[0].shape[0]
        total = levi * bs
        return total, levi.detach()

    # ------------------------------------------------------------------
    # Target builder (identical logic to ComputeGaussianBboxLoss)
    # ------------------------------------------------------------------

    def _build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(7, device=self.device)
        ai   = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        g   = 0.5
        off = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
            device=self.device
        ).float() * g

        for i in range(self.nl):
            anchors_i, shape = self.anchors[i], p[i].shape
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
                j    = torch.stack((torch.ones_like(j), j, k, l, m))
                t    = t.repeat(5, 1, 1)[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t       = targets[0]
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
