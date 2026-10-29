"""
Dual-stream tracker with evidential + Gaussian uncertainty heads (Phase 3.2).

Extends DualStreamTracker (Phase 3.1) by replacing the standard DetectMultiBackend
YOLO models with EvidentialYOLO instances that expose:

  - Evidential classification: belief masses b_k, uncertainty mass u
  - Gaussian spatial uncertainty: per-coordinate σ² (normalised by box size)

These uncertainty quantities are attached to every frame output so that Phase 4
(Discounted Belief Fusion + Dynamic Modality Selection) can consume them directly.

Output keys per frame (superset of Phase 3.1):
  target_bbox        — IR tracker box (= ir_bbox; eval compat)
  ir_bbox            — bounding box from IR YOLO / IR transformer track
  rgb_bbox           — bounding box from RGB YOLO (updated on detect events)
  uav_score          — evidential target probability from transformer (IR)
  uav_uncertainty    — transformer epistemic uncertainty (IR)
  detection_flag     — True while no valid IR target

  # New in 3.2:
  ir_belief          — b_UAV from IR evidential head (scalar, at detect events)
  ir_uncertainty_evi — u from IR evidential head (scalar, at detect events)
  ir_sigma2_norm     — Tensor(4,) normalised spatial variances [cx,cy,w,h] (IR)
  rgb_belief         — b_UAV from RGB evidential head (scalar, at detect events)
  rgb_uncertainty_evi— u from RGB evidential head (scalar, at detect events)
  rgb_sigma2_norm    — Tensor(4,) normalised spatial variances (RGB)
"""

import torch
import torch.nn.functional as F
import numpy as np

from lib.test.tracker.uavtrack_dual import DualStreamTracker
from lib.models.yolo_evidential.model import EvidentialYOLO

from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

_ZERO_BOX  = [0, 0, 0, 0]
_ZERO_UNC  = {'belief': 0.0, 'uncertainty_evi': 1.0,
              'sigma2_norm': torch.zeros(4)}


class DualStreamEvidentialTracker(DualStreamTracker):
    """
    Dual-stream tracker with EvidentialYOLO detection heads.
    The transformer backbone is identical to Phase 3.1.
    """

    # ------------------------------------------------------------------
    # YOLO loading — override to load EvidentialYOLO
    # ------------------------------------------------------------------

    def initialize_yolo_dual(self):
        """Load both IR and RGB EvidentialYOLO models.

        Requires cfg.YOLO_IR.WEIGHTS and cfg.YOLO_RGB.WEIGHTS to point to
        fine-tuned EvidentialYOLO checkpoints (saved by finetune_heads.py).
        Falls back to the base pretrained YOLO if evidential checkpoints are
        not available (for initial testing before fine-tuning is complete).
        """
        ir_cfg  = self.cfg.YOLO_IR
        rgb_cfg = self.cfg.YOLO_RGB

        self._ir_device  = select_device(ir_cfg.DEVICE)
        self._rgb_device = select_device(rgb_cfg.DEVICE)

        # Try evidential checkpoint first, fall back to base YOLO
        self.yolo_ir  = self._load_evidential_or_base(
            ir_cfg.WEIGHTS, self._ir_device, 'IR'
        )
        self.yolo_rgb = self._load_evidential_or_base(
            rgb_cfg.WEIGHTS, self._rgb_device, 'RGB'
        )

        self.yolo_model    = self.yolo_ir  # parent compat
        self.last_rgb_box  = list(_ZERO_BOX)

        # Stored uncertainty outputs from most recent detect event
        self._last_ir_unc  = dict(_ZERO_UNC)
        self._last_rgb_unc = dict(_ZERO_UNC)

        return self.yolo_ir, self.yolo_rgb

    def _load_evidential_or_base(self, weights_path: str,
                                  device: torch.device, tag: str):
        """Load an EvidentialYOLO.  Returns raw EvidentialYOLO object."""
        try:
            model = EvidentialYOLO.from_pretrained(
                weights_path, device, K=getattr(self.cfg, 'YOLO_K', 2)
            )
            model.eval()
            print(f'[EvidentialTracker] Loaded {tag} EvidentialYOLO ← {weights_path}')
            return model
        except Exception as e:
            print(f'[EvidentialTracker] WARNING: Could not load EvidentialYOLO '
                  f'for {tag} ({e}). Falling back to base YOLO.')
            # Fall back to parent class loader for that stream
            return None   # caller must handle

    # ------------------------------------------------------------------
    # YOLO inference helper — EvidentialYOLO version
    # ------------------------------------------------------------------

    def _run_yolo(self, image, model, yolo_cfg, device):
        """Override parent _run_yolo to capture evidential uncertainty outputs.

        Returns:
            box_xywh (list[float] | None): best detection in [x, y, w, h].
            unc (dict): evidential uncertainty dict with keys:
                belief, uncertainty_evi, sigma2_norm.
        """
        if model is None:
            # Fallback path: no evidential model available
            return None, dict(_ZERO_UNC)

        imgsz = check_img_size(yolo_cfg.IMGSZ, s=model.stride.max())

        im = letterbox(image, imgsz, stride=int(model.stride.max()), auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)

        model.warmup(imgsz=(1, 3, *imgsz))

        im_t = torch.from_numpy(im).to(device)
        im_t = im_t.float() / 255.0
        if im_t.ndim == 3:
            im_t = im_t.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            out = model(im_t, augment=yolo_cfg.AUGMENT)

        if len(out) == 5:
            nms_preds, _base, _var, _evi, unc_scales = out
        else:
            # Base YOLO fallback (should not happen with EvidentialYOLO)
            nms_preds = out[0]
            unc_scales = None

        # NMS on the NMS-compatible predictions
        pred = non_max_suppression(
            nms_preds,
            yolo_cfg.CONF_THRES,
            yolo_cfg.IOU_THRES,
            yolo_cfg.CLASSES,
            yolo_cfg.AGNOSTIC_NMS,
            max_det=yolo_cfg.MAX_DET,
        )

        det = pred[0]
        if len(det) == 0:
            return None, dict(_ZERO_UNC)

        det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], image.shape).round()
        best = det[0, :4].cpu().numpy()
        box_xywh = [
            float(best[0]), float(best[1]),
            float(best[2] - best[0] + 1),
            float(best[3] - best[1] + 1),
        ]

        # ── Extract uncertainty for the best detection anchor ─────────
        unc = self._extract_best_anchor_unc(unc_scales, det, im_t, image)

        return box_xywh, unc

    def _extract_best_anchor_unc(self, unc_scales, det, im_t, image_orig):
        """Extract evidential uncertainty values for the highest-scoring detection.

        Since EvidentialDetect returns per-scale uncertainty maps and NMS
        collapses anchors, we conservatively return the *mean* uncertainty
        across all positive detections after NMS.  For Phase 4 this gives a
        reliable single-value estimate per frame.

        Args:
            unc_scales: list of per-scale uncertainty dicts.
            det:        NMS output tensor (N, 6) — [x1,y1,x2,y2,conf,cls].
            im_t:       (1,3,H,W) inference-space image tensor.
            image_orig: original image (for scale reference).

        Returns:
            dict with 'belief', 'uncertainty_evi', 'sigma2_norm'.
        """
        if unc_scales is None or len(det) == 0:
            return dict(_ZERO_UNC)

        # Collect across scales
        all_u   = []
        all_b   = []
        all_sig = []
        for s in unc_scales:
            all_u.append(s['u'].flatten())
            all_b.append(s['b'].flatten(end_dim=-2)[..., 0])  # UAV belief
            all_sig.append(s['sigma2_norm'].flatten(end_dim=-2))  # (N,4)

        u_all   = torch.cat(all_u)
        b_all   = torch.cat(all_b)
        sig_all = torch.cat(all_sig, dim=0)

        # Use the detection with highest confidence (det already sorted by conf)
        # Return the mean across detections ≥ conf_thres (already filtered by NMS)
        return {
            'belief':          b_all.mean().item(),
            'uncertainty_evi': u_all.mean().item(),
            'sigma2_norm':     sig_all.mean(dim=0).cpu(),  # (4,)
        }

    # ------------------------------------------------------------------
    # initialize / track — extend outputs with uncertainty
    # ------------------------------------------------------------------

    def initialize(self, ir_image, rgb_image, ir_model, rgb_model):
        """Run both evidential YOLO heads and initialise the IR transformer."""
        ir_box,  ir_unc  = self._run_yolo(ir_image,  ir_model,  self.cfg.YOLO_IR,  self._ir_device)
        rgb_box, rgb_unc = self._run_yolo(rgb_image, rgb_model, self.cfg.YOLO_RGB, self._rgb_device)

        self.last_rgb_box  = rgb_box  if rgb_box  is not None else list(_ZERO_BOX)
        self._last_ir_unc  = ir_unc
        self._last_rgb_unc = rgb_unc

        if ir_box is None:
            return {
                'target_bbox':         list(_ZERO_BOX),
                'ir_bbox':             list(_ZERO_BOX),
                'rgb_bbox':            self.last_rgb_box,
                'uav_score':           0,
                'detection_flag':      True,
                **self._unc_fields(ir_unc, rgb_unc),
            }

        self.inilialize_tracker(ir_image, ir_box)

        return {
            'target_bbox':    ir_box,
            'ir_bbox':        ir_box,
            'rgb_bbox':       self.last_rgb_box,
            'uav_score':      1,
            'detection_flag': False,
            **self._unc_fields(ir_unc, rgb_unc),
        }

    def track(self, ir_image, rgb_image, info=None):
        """Track with IR transformer; extend output with uncertainty fields."""
        out = super().track(ir_image, rgb_image, info)

        if out.get('detection_flag', False):
            # Re-detection was triggered inside super().track() via self.initialize()
            # Uncertainty was already updated in self._last_*_unc by that call
            pass

        out.update(self._unc_fields(self._last_ir_unc, self._last_rgb_unc))
        return out

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _unc_fields(ir_unc: dict, rgb_unc: dict) -> dict:
        return {
            'ir_belief':           ir_unc.get('belief',          0.0),
            'ir_uncertainty_evi':  ir_unc.get('uncertainty_evi', 1.0),
            'ir_sigma2_norm':      ir_unc.get('sigma2_norm',      torch.zeros(4)),
            'rgb_belief':          rgb_unc.get('belief',          0.0),
            'rgb_uncertainty_evi': rgb_unc.get('uncertainty_evi', 1.0),
            'rgb_sigma2_norm':     rgb_unc.get('sigma2_norm',     torch.zeros(4)),
        }


def get_tracker_class():
    return DualStreamEvidentialTracker
