"""
EvidentialYOLO — wrapper model for Phase 3.2.

Loads a pretrained YOLOv5 checkpoint and replaces the standard ``Detect`` head
with ``EvidentialDetect``, preserving the backbone intact.

Usage::
    model = EvidentialYOLO.from_pretrained(
        weights='/path/to/best_clean.pt',
        device=torch.device('cuda:0'),
        K=2,
    )
    # Fine-tune only the new branches:
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.new_params(), lr=1e-3)

    # Inference:
    nms_preds, _, _, _, unc = model(img_tensor)
    # nms_preds  : (bs, anchors, 6)  — feed to non_max_suppression()
    # unc        : list[dict]  — per-scale uncertainty dicts

Saving / loading::
    model.save('/path/to/evidential_ir.pt')
    model = EvidentialYOLO.load('/path/to/evidential_ir.pt', device)
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Ensure yolov5 is importable when this module is used from the EDTC root
_edtc_root = Path(__file__).resolve().parents[3]
_yolo_root = _edtc_root / 'yolov5'
for p in [str(_edtc_root), str(_yolo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device

from lib.models.yolo_evidential.detect_head import EvidentialDetect


class EvidentialYOLO(nn.Module):
    """YOLOv5 backbone + EvidentialDetect head.

    The backbone is the full pretrained YOLOv5 feature extractor.  Only the
    final ``Detect`` layer is replaced.

    Attributes:
        model:  nn.Sequential — full YOLOv5 model with EvidentialDetect as
                the last layer.
        stride: stride tensor (required by non_max_suppression callers).
        names:  class name list (kept from original).
        K:      number of evidential classes.
    """

    def __init__(self, yolo_model: nn.Module, K: int = 2):
        """Build from an already-loaded DetectionModel.

        Args:
            yolo_model: DetectionModel loaded via attempt_load.
            K:          number of evidential classes.
        """
        super().__init__()

        # Replace the last layer (Detect) with EvidentialDetect
        pretrained_detect = yolo_model.model[-1]
        ev_detect = EvidentialDetect(pretrained_detect, K=K)

        yolo_model.model[-1] = ev_detect
        self.model  = yolo_model.model   # nn.Sequential
        self.stride = yolo_model.stride
        self.names  = yolo_model.names
        self.K      = K

        # Store the YOLOv5 layer-index save list under a distinct name to
        # avoid collision with the save() checkpoint method defined below.
        self._layer_save = getattr(yolo_model, 'save', [])

        # Hyperparameters dict (may be needed by loss function)
        self.hyp = getattr(yolo_model, 'hyp', {})

    # ------------------------------------------------------------------
    # Class methods: build from pretrained checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, weights: str, device: torch.device,
                        K: int = 2, fuse: bool = True) -> 'EvidentialYOLO':
        """Load a YOLOv5 .pt checkpoint and wrap it with EvidentialDetect.

        Args:
            weights: path to pretrained .pt file (e.g. best_clean.pt).
            device:  torch.device to load onto.
            K:       number of evidential classes.
            fuse:    whether to fuse Conv+BN layers before wrapping.

        Returns:
            EvidentialYOLO instance, in eval mode.
        """
        yolo = attempt_load(weights, device=device, inplace=True, fuse=fuse)
        model = cls(yolo, K=K)
        model.to(device)
        return model

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save the full model state (backbone + new heads) to a .pt file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'model_state': self.state_dict(),
            'stride':      self.stride,
            'names':       self.names,
            'K':           self.K,
            'hyp':         self.hyp,
        }, path)
        print(f'[EvidentialYOLO] Saved → {path}')

    @classmethod
    def load(cls, path: str, pretrained_weights: str,
             device: torch.device, K: int = 2) -> 'EvidentialYOLO':
        """Restore an EvidentialYOLO from a saved checkpoint.

        Because EvidentialDetect stores the backbone's Conv2d modules as
        ``self.m`` (not re-parameterised), we first rebuild from the pretrained
        backbone, then load the saved state_dict (which includes the fine-tuned
        m_var / m_evi weights).

        Args:
            path:              path to the saved EvidentialYOLO checkpoint.
            pretrained_weights: path to the original YOLOv5 .pt (for architecture).
            device:            torch.device.
            K:                 number of evidential classes.
        """
        model = cls.from_pretrained(pretrained_weights, device, K=K)
        ckpt  = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state'], strict=True)
        print(f'[EvidentialYOLO] Loaded ← {path}')
        return model

    # ------------------------------------------------------------------
    # Parameter groups for optimisation
    # ------------------------------------------------------------------

    def freeze_backbone(self):
        """Freeze all parameters except the two new head branches.

        Call this before creating the optimiser if you want to train
        only m_var and m_evi while keeping the pretrained backbone fixed.
        """
        head = self.model[-1]   # EvidentialDetect
        # Freeze everything
        for p in self.parameters():
            p.requires_grad_(False)
        # Unfreeze only the new branches
        for p in head.m_var.parameters():
            p.requires_grad_(True)
        for p in head.m_evi.parameters():
            p.requires_grad_(True)
        n_frozen  = sum(not p.requires_grad for p in self.parameters())
        n_trained = sum(p.requires_grad     for p in self.parameters())
        print(f'[EvidentialYOLO] Frozen {n_frozen} params; '
              f'training {n_trained} params (m_var + m_evi only).')

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for p in self.parameters():
            p.requires_grad_(True)

    def new_params(self):
        """Return only the parameters of the new branches (m_var, m_evi)."""
        head = self.model[-1]
        return list(head.m_var.parameters()) + list(head.m_evi.parameters())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, augment: bool = False, profile: bool = False,
                visualize: bool = False):
        """Forward pass through backbone + EvidentialDetect.

        During training:
            Returns (base_raw, var_raw, evi_raw) — see EvidentialDetect.forward.
        During inference:
            Returns (nms_preds, base_raw, var_raw, evi_raw, unc_scales)
            where nms_preds is compatible with non_max_suppression().
        """
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f
                ]
            x = m(x)
            y.append(x if m.i in self._layer_save else None)
        return x

    # ------------------------------------------------------------------
    # Compatibility shims expected by non_max_suppression callers
    # ------------------------------------------------------------------

    @property
    def fp16(self):
        return next(self.parameters()).dtype == torch.float16

    @property
    def pt(self):
        return True

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Run a dummy forward to warm up CUDA kernels."""
        if next(self.parameters()).device.type != 'cpu':
            im = torch.zeros(*imgsz, device=next(self.parameters()).device,
                             dtype=torch.float16 if self.fp16 else torch.float32)
            self.eval()
            with torch.no_grad():
                self(im)

    # ------------------------------------------------------------------
    # Staged-protocol freeze methods (Phase 3.3 revised training)
    # ------------------------------------------------------------------

    def freeze_backbone_train_gaussian(self):
        """Stage 1 Phase A: freeze everything, leave only m_var trainable.

        Use when warming up the Gaussian bbox head before any end-to-end
        fine-tuning.  m_evi is never touched during Stage 1.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        head = self.model[-1]   # EvidentialDetect
        for p in head.m_var.parameters():
            p.requires_grad_(True)
        n_frozen  = sum(not p.requires_grad for p in self.parameters())
        n_trained = sum(p.requires_grad     for p in self.parameters())
        print(f'[EvidentialYOLO] Frozen {n_frozen} params; '
              f'training {n_trained} params (m_var only — Stage 1 Phase A).')

    def freeze_evidential_head(self):
        """Stage 1 Phase B: re-freeze m_evi after unfreeze_all().

        Call this immediately after unfreeze_all() when transitioning to
        Phase B so that m_evi stays fixed throughout Stage 1.
        """
        head = self.model[-1]
        for p in head.m_evi.parameters():
            p.requires_grad_(False)
        n_trained = sum(p.requires_grad for p in self.parameters())
        print(f'[EvidentialYOLO] m_evi frozen; '
              f'training {n_trained} params (backbone+neck+m+m_var — Stage 1 Phase B).')

    def freeze_all_train_evidential(self):
        """Stage 2: freeze backbone, neck, m, m_var — only m_evi trains.

        The entire Stage 1 network is frozen so EDL gradients cannot corrupt
        the Gaussian representations learned in Stage 1.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        head = self.model[-1]
        for p in head.m_evi.parameters():
            p.requires_grad_(True)
        n_frozen  = sum(not p.requires_grad for p in self.parameters())
        n_trained = sum(p.requires_grad     for p in self.parameters())
        print(f'[EvidentialYOLO] Frozen {n_frozen} params; '
              f'training {n_trained} params (m_evi only — Stage 2).')

    def gaussian_params(self):
        """Return only the m_var (log-variance) branch parameters."""
        return list(self.model[-1].m_var.parameters())

    def evidential_params(self):
        """Return only the m_evi (Dirichlet evidence) branch parameters."""
        return list(self.model[-1].m_evi.parameters())
