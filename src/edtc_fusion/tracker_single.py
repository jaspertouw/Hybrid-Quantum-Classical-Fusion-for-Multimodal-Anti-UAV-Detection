"""
tracker_single.py — Single-modality EDTC baseline
=================================================
EDTCSingleTracker — one-stream uncertainty-aware tracker that mirrors the
fusion tracker (YOLO -> SiamCAR with two-state switching) but operates on
a SINGLE modality (IR or RGB).  No DBF/ABF, no DMS — the switching signal
is the per-modality vacuity u and the per-box-normalised spatial variance
sigma2.  This is the thermal-only / RGB-only baseline against which the
DBF and ABF fusion runs are compared.

State machine (matches tracker.py)
----------------------------------
    DETECT : run YOLO. If  u < theta_cls_detect  AND  sigma2 < theta_loc_detect
             init SiamCAR with the YOLO box and transition to TRACK.
    TRACK  : run SiamCAR. If NOT (u < theta_cls_track AND sigma2 < theta_loc_track)
             revert to DETECT.

sigma2 normalisation matches what DMS does internally for fusion (variance
divided by [w, h, w, h] then mean) so the threshold scale is comparable to
sigma2_best in the fusion runs.

The implementation reuses the YOLO/SiamCAR plumbing from tracker.py to avoid
duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.edtc_fusion.tracker import (
    _ZERO_UNC,
    _EvidentialSiamCARTracker,
    _read_hp,
    _yolo_run,
)
from yolov5.utils.general import check_img_size

# Re-import the same modules tracker.py bootstrapped via sys.path
from pysot.core.config import cfg as siam_cfg  # noqa: E402
from pysot.models.siamcar_gaussian import SiamCARStage2  # noqa: E402
from lib.models.yolo_evidential.model import EvidentialYOLO  # noqa: E402


_DMS_EPS = 1e-8


def _normalise_sigma2(sigma2: torch.Tensor, box_xywh) -> float:
    """Divide raw sigma2 (cx,cy,w,h) by [w, h, w, h] then mean — matches the
    per-box-size normalisation that DMS applies to sigma2_best in the fusion
    pipeline, so thresholds stay on the same scale across runs."""
    w = max(float(box_xywh[2]), _DMS_EPS)
    h = max(float(box_xywh[3]), _DMS_EPS)
    scale = torch.tensor([w, h, w, h], dtype=sigma2.dtype)
    return float((sigma2 / scale).mean().item())


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class SingleConfig:
    """Knobs for a single-modality EDTC baseline run.

    Mirrors FusionConfig but only carries paths for one modality.  Same four
    switching thresholds as the fusion config — for a one-stream signal the
    threshold scale should still be sensible because:
      * YOLO u is a mean over anchors (same scale per modality).
      * sigma2 is per-box-normalised here, matching sigma2_best in fusion.
    Phase-6.2 grid search will refine per modality.
    """
    # Modality selector
    modality: str = "ir"            # "ir" or "rgb"

    # YOLO (one modality)
    yolo_weights: str = ""
    yolo_base:    str = ""

    # SiamCAR Stage-2 (one modality)
    siam_stage2:     str = ""
    siam_stage1:     str = ""
    siam_pretrained: str = ""
    siam_cfg:        str = ""

    K: int = 2

    # YOLO uncertainty aggregation, read by _yolo_run (see FusionConfig.unc_agg):
    #   "mean" — average over all pre-NMS anchors (legacy default)
    #   "peak" — uncertainty of the selected detection only
    unc_agg: str = "mean"

    # Switching thresholds — same defaults as the fusion config
    theta_cls_detect: float = 0.80
    theta_loc_detect: float = 0.10
    theta_cls_track:  float = 0.90
    theta_loc_track:  float = 0.70

    # YOLO inference
    yolo_imgsz:      tuple = (640, 640)
    yolo_conf_thres: float = 0.25
    yolo_iou_thres:  float = 0.45
    yolo_max_det:    int   = 300
    yolo_classes:    list  = None    # set in __post_init__

    # SiamCAR HP (lr, penalty_k, window_lr) — pulled from YAML if None
    hp: tuple | None = None

    device: str = "cuda:0"

    def __post_init__(self):
        if self.yolo_classes is None:
            self.yolo_classes = [0]
        if self.modality not in ("ir", "rgb"):
            raise ValueError(f"modality must be 'ir' or 'rgb', got {self.modality!r}")


# ===========================================================================
# Tracker
# ===========================================================================

class EDTCSingleTracker:
    """Single-modality YOLO+SiamCAR tracker with EDTC two-state switching."""

    DETECT = 0
    TRACK  = 1

    def __init__(self, config: SingleConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.state = self.DETECT

        # YOLO
        self.yolo = EvidentialYOLO.load(
            config.yolo_weights, config.yolo_base, self.device, K=config.K,
        )
        self.yolo.eval()
        self._yolo_stride = int(self.yolo.stride.max())
        self._yolo_imgsz  = check_img_size(config.yolo_imgsz, s=self._yolo_stride)

        # SiamCAR — populate the global yacs cfg from this modality's YAML
        siam_cfg.merge_from_file(config.siam_cfg)
        self.siam_model = SiamCARStage2.load(
            config.siam_stage2,
            stage1_path=config.siam_stage1,
            pretrained_path=config.siam_pretrained,
            K=config.K,
        ).to(self.device).eval()
        self.siam_tracker = _EvidentialSiamCARTracker(self.siam_model, siam_cfg.TRACK)

        # HP_SEARCH key matches the YAML convention used in the fusion config
        hp_key = "ANTIUAVTHERMAL" if config.modality == "ir" else "ANTIUAVRGB"
        self.hp = config.hp or _read_hp(config.siam_cfg, hp_key)

        self._initialized = False
        self.frame_log = []

    # ------------------------------------------------------------------

    def reset(self):
        self.state = self.DETECT
        self._initialized = False
        self.frame_log = []

    def step(self, frame_bgr: np.ndarray) -> dict:
        if self.state == self.DETECT:
            return self._detect_step(frame_bgr)
        return self._track_step(frame_bgr)

    # ------------------------------------------------------------------

    def _detect_step(self, frame_bgr) -> dict:
        box, unc = _yolo_run(frame_bgr, self.yolo, self.cfg, self.device,
                             self._yolo_imgsz, self._yolo_stride)

        if box is None:
            self._log("DETECT_NONE", [0.0, 0.0, 0.0, 0.0], 1.0, float("inf"), False, b_uav=0.0)
            return {"bbox": [0.0, 0.0, 0.0, 0.0], "state": "DETECT",
                    "u": 1.0, "sigma2": float("inf"), "switched": False}

        u_f     = float(unc["u"].item())
        b_uav_f = float(unc["b"][0].item())
        b_bg_f  = float(unc["b"][1].item())
        s_f = _normalise_sigma2(unc["sigma2_norm"], box)

        switched = False
        if b_uav_f > b_bg_f and u_f < self.cfg.theta_cls_detect and s_f < self.cfg.theta_loc_detect:
            self.siam_tracker.init(frame_bgr, box)
            self._initialized = True
            self.state = self.TRACK
            switched = True

        self._log("DETECT", box, u_f, s_f, switched, b_uav=b_uav_f)
        return {"bbox": box, "state": "DETECT",
                "u": u_f, "b_uav": b_uav_f, "sigma2": s_f, "switched": switched}

    def _track_step(self, frame_bgr) -> dict:
        hp = {"lr": self.hp[0], "penalty_k": self.hp[1], "window_lr": self.hp[2]}
        with torch.no_grad():
            out = self.siam_tracker.track(frame_bgr, hp)

        u_f     = float(out["u"].item())
        b_uav_f = float(out["b"][0].item())
        b_bg_f  = float(out["b"][1].item())
        s_f = _normalise_sigma2(out["sigma2_norm"], out["bbox"])
        box = out["bbox"]

        keep = (b_uav_f > b_bg_f and u_f < self.cfg.theta_cls_track and s_f < self.cfg.theta_loc_track)
        if not keep:
            self.state = self.DETECT

        self._log("TRACK", box, u_f, s_f, not keep, b_uav=b_uav_f)
        return {"bbox": box, "state": "TRACK",
                "u": u_f, "b_uav": b_uav_f, "sigma2": s_f, "switched": (not keep)}

    # ------------------------------------------------------------------

    def _log(self, mode, box, u, s, switched, b_uav=None):
        self.frame_log.append({
            "state":    mode,
            "bbox":     box,
            "u":        u,
            "b_uav":    b_uav,
            "sigma2":   s,
            "switched": switched,
        })
