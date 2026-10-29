import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.models.head.car_head import CARHead
from pysot.utils.model_load import load_pretrain
from pysot.utils.xcorr import xcorr_depthwise
from pysot.utils.location_grid import compute_locations


class CARHeadGaussian(CARHead):
    """CARHead that also returns bbox_tower features for the variance branch."""

    def forward(self, x):
        cls_tower_feat = self.cls_tower(x)
        logits = self.cls_logits(cls_tower_feat)
        centerness = self.centerness(cls_tower_feat)
        bbox_tower_feat = self.bbox_tower(x)
        bbox_reg = torch.exp(self.bbox_pred(bbox_tower_feat))
        return logits, bbox_reg, centerness, bbox_tower_feat


class SiamCARGaussian(nn.Module):
    """
    SiamCAR + parallel Gaussian log-variance head for bounding box uncertainty.

    Architecture:
        - base: frozen/partially-frozen ModelBuilder (backbone, neck, xcorr, down, CARHeadGaussian)
        - bbox_var: trainable Conv2d(256, 4) predicting log σ² for [l, t, r, b]

    Training stages:
        Phase A (warm-up): freeze_base() — only bbox_var trains
        Phase B (joint):   freeze_backbone_only() — neck+head+down+bbox_var train
    """

    def __init__(self, pretrained_path: str, in_channels: int = 256):
        super().__init__()

        base = ModelBuilder()
        load_pretrain(base, pretrained_path)

        # Swap car_head for the subclass that exposes bbox_tower features.
        # CARHeadGaussian shares identical parameters with CARHead, so strict=True works.
        extended_head = CARHeadGaussian(cfg, in_channels)
        extended_head.load_state_dict(base.car_head.state_dict(), strict=True)
        base.car_head = extended_head

        self.base = base

        # New trainable branch: log-variance for ltrb coordinates.
        # Bias init 4.0 → σ²≈55, reasonable for ltrb distances of 50-100 px.
        # Avoids the initial NLL explosion that bias=0 (σ²=1) would cause.
        self.bbox_var = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.bbox_var.weight, std=0.01)
        nn.init.constant_(self.bbox_var.bias, 4.0)

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_base(self):
        """Phase A: freeze entire base, train only bbox_var."""
        for p in self.base.parameters():
            p.requires_grad_(False)
        for m in self.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for p in self.bbox_var.parameters():
            p.requires_grad_(True)
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[SiamCARGaussian] Phase A: base frozen, bbox_var trainable '
              f'({n_trainable} params)')

    def freeze_backbone_only(self):
        """Phase B: freeze backbone, train neck+car_head+down+bbox_var."""
        for p in self.base.backbone.parameters():
            p.requires_grad_(False)
        for m in self.base.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for p in self.base.neck.parameters():
            p.requires_grad_(True)
        for p in self.base.car_head.parameters():
            p.requires_grad_(True)
        for p in self.base.down.parameters():
            p.requires_grad_(True)
        for p in self.bbox_var.parameters():
            p.requires_grad_(True)
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[SiamCARGaussian] Phase B: backbone frozen, '
              f'neck+head+down+bbox_var trainable ({n_trainable} params)')

    def gaussian_params(self):
        """Return only bbox_var parameters (for Phase A optimizer)."""
        return list(self.bbox_var.parameters())

    # ------------------------------------------------------------------
    # Feature extraction shared between forward and track
    # ------------------------------------------------------------------

    def _extract_features(self, template, search):
        zf = self.base.backbone(template)
        xf = self.base.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.base.neck(zf)
            xf = self.base.neck(xf)
        features = xcorr_depthwise(xf[0], zf[0])
        for i in range(len(xf) - 1):
            features = torch.cat([features, xcorr_depthwise(xf[i + 1], zf[i + 1])], dim=1)
        return self.base.down(features)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, data):
        """
        Returns dict with cls (raw logits), loc (post-exp ltrb), cen,
        log_var (B,4,H,W), locations — labels NOT consumed here.
        """
        template = data['template'].cuda()
        search   = data['search'].cuda()

        features = self._extract_features(template, search)
        cls, loc, cen, bbox_tower_feat = self.base.car_head(features)
        log_var   = self.bbox_var(bbox_tower_feat)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)

        return {
            'cls':       cls,
            'loc':       loc,
            'cen':       cen,
            'log_var':   log_var,
            'locations': locations,
        }

    # ------------------------------------------------------------------
    # Inference (tracking)
    # ------------------------------------------------------------------

    def template(self, z):
        zf = self.base.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.base.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.base.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.base.neck(xf)
        features = xcorr_depthwise(xf[0], self.zf[0])
        for i in range(len(xf) - 1):
            features = torch.cat(
                [features, xcorr_depthwise(xf[i + 1], self.zf[i + 1])], dim=1
            )
        features = self.base.down(features)
        cls, loc, cen, bbox_tower_feat = self.base.car_head(features)
        log_var = self.bbox_var(bbox_tower_feat)
        return {'cls': cls, 'loc': loc, 'cen': cen, 'log_var': log_var}

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({'model_state': self.state_dict()}, path)
        print(f'[SiamCARGaussian] Saved → {path}')

    @classmethod
    def load(cls, path: str, pretrained_path: str, in_channels: int = 256):
        model = cls(pretrained_path, in_channels)
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
        # Strip 'module.' prefix if saved from DataParallel
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        print(f'[SiamCARGaussian] Loaded ← {path}')
        return model
