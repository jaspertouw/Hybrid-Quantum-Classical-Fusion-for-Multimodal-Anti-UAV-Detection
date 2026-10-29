import torch
import torch.nn as nn

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.models.head.car_head import CARHead
from pysot.utils.xcorr import xcorr_depthwise
from pysot.utils.location_grid import compute_locations


class CARHeadFull(CARHead):
    """CARHead that exposes both bbox_tower and cls_tower features.

    Subclasses CARHead directly (not CARHeadGaussian) to avoid a cross-file
    dependency. Parameter names are identical to CARHead, so a Stage 1
    checkpoint loads with strict=True.
    """

    def forward(self, x):
        cls_tower_feat  = self.cls_tower(x)
        logits          = self.cls_logits(cls_tower_feat)
        centerness      = self.centerness(cls_tower_feat)
        bbox_tower_feat = self.bbox_tower(x)
        bbox_reg        = torch.exp(self.bbox_pred(bbox_tower_feat))
        return logits, bbox_reg, centerness, bbox_tower_feat, cls_tower_feat


class SiamCARStage2(nn.Module):
    """Stage 2 wrapper: Stage 1 network + parallel evidential classification head.

    Architecture:
        base.backbone + neck + down + CARHeadFull  — all frozen
        bbox_var  Conv2d(256, 4, 3, p=1)           — frozen (Stage 1 branch)
        cls_evi   Conv2d(256, K, 3, p=1)           — trainable (new)

    At inference, cls_evi replaces cls_logits as the classification signal,
    producing Dirichlet evidence for the downstream uncertainty fusion module.

    Args:
        stage1_path:      path to Stage 1 _best.pth or _last.pth
        pretrained_path:  original SiamCAR baseline .pth (for arch initialisation)
        K:                number of evidential classes (default 2: bg / UAV)
        in_channels:      feature channels entering cls_evi (default 256)
    """

    def __init__(self,
                 stage1_path: str,
                 pretrained_path: str,
                 K: int = 2,
                 in_channels: int = 256):
        super().__init__()

        self.K = K

        # Build architecture (weights come from Stage 1 checkpoint below).
        base = ModelBuilder()

        # Swap car_head for CARHeadFull which returns cls_tower_feat too.
        # CARHeadFull has identical parameter names to CARHead → strict=True works.
        full_head = CARHeadFull(cfg, in_channels)
        full_head.load_state_dict(base.car_head.state_dict(), strict=True)
        base.car_head = full_head

        self.base = base

        # Stage 1 branch — placeholder; overwritten by _load_stage1 below.
        self.bbox_var = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)

        # New trainable branch: raw evidence logits for K classes.
        # bias=0 → symmetric start; ReLU+1 initialises α≈1 (uniform Dirichlet).
        self.cls_evi = nn.Conv2d(in_channels, K, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.cls_evi.weight, std=0.01)
        nn.init.constant_(self.cls_evi.bias, 0.0)

        self._load_stage1(stage1_path)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_stage1(self, stage1_path: str) -> None:
        """Load Stage 1 weights; assert only cls_evi keys are missing."""
        ckpt  = torch.load(stage1_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt.get('model_state', ckpt))
        # Strip DataParallel prefix if present.
        state = {k.replace('module.', '', 1): v for k, v in state.items()}

        own_keys  = set(self.state_dict().keys())
        ckpt_keys = set(state.keys())
        missing     = own_keys  - ckpt_keys   # should be exactly cls_evi.{weight,bias}
        unexpected  = ckpt_keys - own_keys     # should be empty

        expected_missing = {f'cls_evi.weight', f'cls_evi.bias'}
        assert missing == expected_missing, (
            f'[SiamCARStage2] Unexpected missing keys when loading Stage 1:\n'
            f'  Got:      {missing}\n'
            f'  Expected: {expected_missing}\n'
            f'  Hint: check that stage1_path is a SiamCARGaussian checkpoint.'
        )
        assert len(unexpected) == 0, (
            f'[SiamCARStage2] Unexpected keys in Stage 1 checkpoint: {unexpected}'
        )

        self.load_state_dict(state, strict=False)
        print(f'[SiamCARStage2] Stage 1 loaded from {stage1_path}')
        print(f'[SiamCARStage2] Missing (expected — new cls_evi): {missing}')

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze_all_train_evidential(self) -> None:
        """Stage 2: freeze everything; train only cls_evi."""
        for p in self.base.parameters():
            p.requires_grad_(False)
        for m in self.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for p in self.bbox_var.parameters():
            p.requires_grad_(False)
        for p in self.cls_evi.parameters():
            p.requires_grad_(True)

        n_frozen   = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[SiamCARStage2] Frozen {n_frozen} params; '
              f'training {n_trainable} params (cls_evi only)')

    def evidential_params(self):
        """Return cls_evi parameters only (for the Stage 2 optimizer)."""
        return list(self.cls_evi.parameters())

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, template, search):
        zf = self.base.backbone(template)
        xf = self.base.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.base.neck(zf)
            xf = self.base.neck(xf)
        features = xcorr_depthwise(xf[0], zf[0])
        for i in range(len(xf) - 1):
            features = torch.cat(
                [features, xcorr_depthwise(xf[i + 1], zf[i + 1])], dim=1
            )
        return self.base.down(features)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(self, data):
        """Training forward. Labels NOT consumed here (injected by training loop).

        Returns dict:
            cls       (B, 2, H, W)  frozen logits (not used for Stage 2 loss)
            loc       (B, 4, H, W)  post-exp ltrb
            cen       (B, 1, H, W)
            log_var   (B, 4, H, W)  frozen Stage 1 variance
            evi_raw   (B, K, H, W)  raw evidence (pre-ReLU)
            locations (H*W, 2)
        """
        template = data['template'].cuda()
        search   = data['search'].cuda()

        features = self._extract_features(template, search)
        cls, loc, cen, bbox_tower_feat, cls_tower_feat = self.base.car_head(features)

        log_var = self.bbox_var(bbox_tower_feat)
        evi_raw = self.cls_evi(cls_tower_feat)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)

        return {
            'cls':       cls,
            'loc':       loc,
            'cen':       cen,
            'log_var':   log_var,
            'evi_raw':   evi_raw,
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
        """Returns cls, loc, cen, log_var, evi_raw.

        Downstream tracker should use evi_raw (after relu+alpha+S→belief masses)
        instead of cls for uncertainty-aware target scoring.
        """
        xf = self.base.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.base.neck(xf)
        features = xcorr_depthwise(xf[0], self.zf[0])
        for i in range(len(xf) - 1):
            features = torch.cat(
                [features, xcorr_depthwise(xf[i + 1], self.zf[i + 1])], dim=1
            )
        features = self.base.down(features)
        cls, loc, cen, bbox_tower_feat, cls_tower_feat = self.base.car_head(features)
        log_var = self.bbox_var(bbox_tower_feat)
        evi_raw = self.cls_evi(cls_tower_feat)
        return {'cls': cls, 'loc': loc, 'cen': cen,
                'log_var': log_var, 'evi_raw': evi_raw}

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({'model_state': self.state_dict(), 'K': self.K}, path)
        print(f'[SiamCARStage2] Saved → {path}')

    @classmethod
    def load(cls,
             path: str,
             stage1_path: str,
             pretrained_path: str,
             K: int = 2,
             in_channels: int = 256) -> 'SiamCARStage2':
        """Restore a saved Stage 2 checkpoint."""
        model = cls(stage1_path, pretrained_path, K=K, in_channels=in_channels)
        ckpt  = torch.load(path, map_location='cpu')
        state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        print(f'[SiamCARStage2] Loaded ← {path}')
        return model
