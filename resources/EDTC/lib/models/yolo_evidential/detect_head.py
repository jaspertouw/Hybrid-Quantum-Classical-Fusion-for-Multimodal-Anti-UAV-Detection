"""
EvidentialDetect — extended YOLOv5 detection head (Phase 3.2).

Extends the standard Detect head with two additional parallel branches:
  - Gaussian bbox branch  : outputs log-variance per coordinate (σ² for spatial uncertainty)
  - Evidential class branch: outputs K=2 evidence values (UAV vs. background)

The pretrained bbox+objectness convolutions are kept intact and can be frozen
during fine-tuning so only the two new branches need to learn from scratch.

Output per anchor during training:
  base : [cx_raw, cy_raw, w_raw, h_raw, obj_raw, cls_raw(ignored)]   shape (..., 6)
  var  : [log_var_cx, log_var_cy, log_var_w, log_var_h]               shape (..., 4)
  evi  : [e_0 (UAV), e_1 (background)]  — raw logits, ReLU applied    shape (..., K)

Output during inference (NMS-compatible):
  A single concatenated tensor identical in structure to the original Detect
  inference output, but with:
    - columns 5+ replaced by the UAV belief mass b_0 = e_0 / S
  This lets the standard non_max_suppression() work unchanged.

Additionally returns a dict of evidential/uncertainty quantities for the
caller (DualStreamTracker) to store and pass forward to Phase 4 fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov5.utils.general import check_version

_TORCH_GE_110 = check_version(torch.__version__, '1.10.0')


class EvidentialDetect(nn.Module):
    """Extended detection head with Gaussian spatial uncertainty and
    evidential classification uncertainty.

    Args:
        pretrained_detect: the ``Detect`` instance from a loaded YOLOv5 model
            (``model.model[-1]`` after ``attempt_load``).
        K: number of evidential classes.  Use K=2 for Anti-UAV (UAV / bg).
    """

    stride = None   # filled in by parent model after first forward
    dynamic = False
    export = False
    use_cls_for_conf = False  # True during Stage 1 val: use pretrained cls logit instead of b_UAV

    def __init__(self, pretrained_detect, K: int = 2):
        super().__init__()
        pd = pretrained_detect

        self.nc = pd.nc          # original class count (1 for Anti-UAV)
        self.no = pd.no          # original outputs per anchor = nc + 5
        self.nl = pd.nl          # number of detection layers (scales)
        self.na = pd.na          # anchors per scale
        self.K = K               # evidential classes
        self.inplace = pd.inplace

        # ── YOLOv5 layer metadata ─────────────────────────────────────────
        # parse_model() attaches i (layer index), f (from index), type, np
        # to every module.  Copy them here so the forward loop in BaseModel
        # (and our EvidentialYOLO.forward) can inspect m.f and m.i.
        for attr in ('i', 'f', 'type', 'np'):
            if hasattr(pd, attr):
                setattr(self, attr, getattr(pd, attr))

        # ── Anchor geometry ──────────────────────────────────────────────
        # Copy anchors and grid buffers from the pretrained head.
        self.register_buffer('anchors', pd.anchors.clone())
        self.stride = pd.stride          # tensor filled in by DetectionModel
        self.grid = pd.grid              # list of empty tensors
        self.anchor_grid = pd.anchor_grid
        self.dynamic = pd.dynamic
        self.export = pd.export

        # ── Branch 1: bbox means + objectness (pretrained) ───────────────
        # Keep the full original Conv2d modules from the pretrained head.
        # During loss computation we use columns 0-4 (bbox+obj); the original
        # class column (5) is ignored in favour of the evidence branch.
        self.m = pd.m   # nn.ModuleList — do NOT replace; keep pretrained weights

        # ── Branch 2: log-variance (new, trained from scratch) ───────────
        # Outputs 4 log-variances per anchor (one per bbox coordinate).
        ch_in = [conv.in_channels for conv in pd.m]
        self.m_var = nn.ModuleList(
            nn.Conv2d(c, 4 * self.na, 1) for c in ch_in
        )

        # ── Branch 3: evidence (new, trained from scratch) ───────────────
        # Outputs K raw evidence logits per anchor.
        self.m_evi = nn.ModuleList(
            nn.Conv2d(c, K * self.na, 1) for c in ch_in
        )

        self._init_new_branches()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_new_branches(self):
        """Initialise the new branches to safe starting values.

        log-variance branch: zero weight, zero bias → σ² = exp(0) = 1 at init,
          giving a uniform-ish variance before any training signal.
        Evidence branch: Xavier uniform weight, small positive bias so that
          ReLU(logit) > 0 at init (avoids dead neurons).
        """
        for m in self.m_var:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        for m in self.m_evi:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.1)

    # ------------------------------------------------------------------
    # Grid helper (identical to original Detect)
    # ------------------------------------------------------------------

    def _make_grid(self, nx: int = 20, ny: int = 20, i: int = 0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y = torch.arange(ny, device=d, dtype=t)
        x = torch.arange(nx, device=d, dtype=t)
        yv, xv = (torch.meshgrid(y, x, indexing='ij')
                  if _TORCH_GE_110 else torch.meshgrid(y, x))
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (
            (self.anchors[i] * self.stride[i])
            .view(1, self.na, 1, 1, 2)
            .expand(shape)
        )
        return grid, anchor_grid

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: list):
        """
        Args:
            x: list of feature maps, one per detection scale.

        Training returns:
            (base_raw, var_raw, evi_raw)
            base_raw : list[Tensor(bs, na, H, W, no)]  — raw bbox+obj+cls
            var_raw  : list[Tensor(bs, na, H, W, 4)]   — raw log-variances
            evi_raw  : list[Tensor(bs, na, H, W, K)]   — raw evidence logits

        Inference returns:
            (nms_preds, base_raw, var_raw, evi_raw, uncertainty_dict)
            nms_preds : Tensor(bs, total_anchors, 6)  — NMS-compatible
                        [cx, cy, w, h, obj_sigmoid, b_UAV]
            uncertainty_dict: dict with per-scale u, b, sigma2_norm
        """
        bs = x[0].shape[0]
        base_raw, var_raw, evi_raw = [], [], []

        for i in range(self.nl):
            feat = x[i]

            # ── Run all three branches ────────────────────────────────
            raw_base = self.m[i](feat)         # (bs, no*na, H, W)
            raw_var  = self.m_var[i](feat)     # (bs, 4*na,  H, W)
            raw_evi  = self.m_evi[i](feat)     # (bs, K*na,  H, W)

            bs, _, ny, nx = raw_base.shape

            # Reshape to (bs, na, H, W, channels)
            raw_base = (raw_base
                        .view(bs, self.na, self.no, ny, nx)
                        .permute(0, 1, 3, 4, 2).contiguous())
            raw_var  = (raw_var
                        .view(bs, self.na, 4, ny, nx)
                        .permute(0, 1, 3, 4, 2).contiguous())
            raw_evi  = (raw_evi
                        .view(bs, self.na, self.K, ny, nx)
                        .permute(0, 1, 3, 4, 2).contiguous())

            base_raw.append(raw_base)
            var_raw.append(raw_var)
            evi_raw.append(raw_evi)

        if self.training:
            return base_raw, var_raw, evi_raw

        # ── Inference ─────────────────────────────────────────────────
        z = []
        unc_scales = []

        for i in range(self.nl):
            rb = base_raw[i]   # (bs, na, H, W, no)
            rv = var_raw[i]    # (bs, na, H, W, 4)
            re = evi_raw[i]    # (bs, na, H, W, K)

            bs, _, ny, nx, _ = rb.shape

            if self.dynamic or self.grid[i].shape[2:4] != (ny, nx):
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            # ── Bbox means (same activations as original Detect) ──────
            xy_raw, wh_raw, obj_raw = rb[..., :2], rb[..., 2:4], rb[..., 4:5]
            xy  = (xy_raw.sigmoid()  * 2 + self.grid[i])  * self.stride[i]
            wh  = (wh_raw.sigmoid()  * 2) ** 2 * self.anchor_grid[i]
            obj = obj_raw.sigmoid()

            # ── Evidence → beliefs + uncertainty ──────────────────────
            # Evidence values must be ≥ 0  (guaranteed by ReLU)
            evidence = F.relu(re)                           # (bs, na, H, W, K)
            alpha    = evidence + 1.0                       # Dirichlet params
            S        = alpha.sum(dim=-1, keepdim=True)      # precision (bs,na,H,W,1)
            b        = evidence / S                         # belief masses
            u        = self.K / S                           # uncertainty mass (∈(0,1])

            # UAV belief (class index 0) replaces original class logit.
            # During Stage 1 validation (m_evi untrained), use the pretrained
            # cls sigmoid instead so NMS confidence is not destroyed.
            b_uav = b[..., 0:1]                             # (bs, na, H, W, 1)
            if self.use_cls_for_conf:
                b_uav = rb[..., 5:6].sigmoid()              # pretrained cls logit

            # ── σ² (from log-variance branch) ─────────────────────────
            log_var = rv.clamp(-10, 10)                     # prevent exp overflow → inf
            sigma2  = torch.exp(log_var)                    # positive variances

            # Scale-normalised variances (He et al. 2019)
            # σ̂²_x = σ²_x / w_pred,  σ̂²_y = σ²_y / h_pred
            # σ̂²_w = σ²_w / w_pred², σ̂²_h = σ²_h / h_pred²
            w_pred = wh[..., 0:1]
            h_pred = wh[..., 1:2]
            sigma2_norm = torch.cat([
                sigma2[..., 0:1] / (w_pred + 1e-6),
                sigma2[..., 1:2] / (h_pred + 1e-6),
                sigma2[..., 2:3] / (w_pred ** 2 + 1e-6),
                sigma2[..., 3:4] / (h_pred ** 2 + 1e-6),
            ], dim=-1)

            unc_scales.append({
                'u':           u,            # (bs, na, H, W, 1)
                'b':           b,            # (bs, na, H, W, K)
                'sigma2':      sigma2,       # (bs, na, H, W, 4)  absolute
                'sigma2_norm': sigma2_norm,  # (bs, na, H, W, 4)  normalised
            })

            # ── NMS-compatible pack: [cx,cy,w,h, obj, b_UAV] ──────────
            y = torch.cat([xy, wh, obj, b_uav], dim=-1)
            z.append(y.view(bs, self.na * nx * ny, 6))

        nms_preds = torch.cat(z, 1)   # (bs, total_anchors, 6)

        if self.export:
            return (nms_preds,)
        return nms_preds, base_raw, var_raw, evi_raw, unc_scales
