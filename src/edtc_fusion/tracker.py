"""
tracker.py — Phase 6.1
======================
EDTCFusionTracker — dual-stream uncertainty-aware tracker with the four-
threshold switching logic from the EDTC fusion design.

Pipeline per frame
------------------
    if state == DETECT:
        # Run both YOLO evidential heads in parallel
        ir_box, (b_ir, u_ir, sigma2_ir)   = yolo_ir(ir_frame)
        rgb_box, (b_rgb, u_rgb, sigma2_rgb) = yolo_rgb(rgb_frame)

        # Belief fusion + Dynamic Modality Selection
        b_fused, u_fused      = dbf(b_rgb, u_rgb, b_ir, u_ir)
        best_box, sigma2_best = dms(box_rgb, var_rgb, box_ir, var_ir)

        # INIT condition (Phase 6.1)
        if u_fused < theta_cls_detect AND sigma2_best < theta_loc_detect:
            siamcar_ir.init(ir_frame,  best_box)
            siamcar_rgb.init(rgb_frame, best_box)
            state = TRACK

    if state == TRACK:
        # Run both SiamCAR Stage-2 heads in parallel
        ir_out  = siamcar_ir.track(ir_frame)
        rgb_out = siamcar_rgb.track(rgb_frame)

        # Extract per-modality uncertainty at each peak
        b_ir, u_ir, sigma2_ir,   box_ir  = peak_uncertainty(ir_out)
        b_rgb, u_rgb, sigma2_rgb, box_rgb = peak_uncertainty(rgb_out)

        # Fusion + DMS
        b_fused, u_fused      = dbf(b_rgb, u_rgb, b_ir, u_ir)
        best_box, sigma2_best = dms(box_rgb, var_rgb, box_ir, var_ir)

        # MAINTAIN condition
        if NOT (u_fused < theta_cls_track AND sigma2_best < theta_loc_track):
            state = DETECT

Class index convention (both detection and tracking branches)
    0 = UAV  (foreground)   1 = background
This matches the convention in belief_fusion.py and dms.py.

The tracker is intentionally written to be standalone and importable from
``src.edtc_fusion`` — it does not subclass any of the existing trackers
in resources/EDTC or resources/SiamCAR.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path bootstrapping — make the EDTC and SiamCAR resources importable.
# Done at import-time so the tracker is usable from anywhere in the repo.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EDTC_ROOT = _REPO_ROOT / "resources" / "EDTC"
_SIAMCAR_ROOT = _REPO_ROOT / "resources" / "SiamCAR"
_YOLO_ROOT = _EDTC_ROOT / "yolov5"

for _p in (str(_EDTC_ROOT), str(_YOLO_ROOT), str(_SIAMCAR_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# EDTC pieces
from lib.utils.belief_fusion import abf, dbf  # noqa: E402
from lib.utils.dms import dms  # noqa: E402
from lib.models.yolo_evidential.model import EvidentialYOLO  # noqa: E402

# YOLOv5 utilities
from yolov5.utils.augmentations import letterbox  # noqa: E402
from yolov5.utils.general import (  # noqa: E402
    check_img_size,
    non_max_suppression,
    scale_boxes,
)

# SiamCAR pieces
from pysot.core.config import cfg as siam_cfg  # noqa: E402
from pysot.models.siamcar_gaussian import SiamCARStage2  # noqa: E402
from pysot.tracker.siamcar_tracker import SiamCARTracker  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (module level)
# ---------------------------------------------------------------------------

def _read_hp(yaml_path: str, key: str) -> tuple:
    """Pull HP_SEARCH.<key> = [lr, penalty_k, window_lr] from a YAML cfg."""
    import yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    hp = data.get("HP_SEARCH", {}).get(key)
    if hp is None:
        raise KeyError(f"HP_SEARCH.{key} not found in {yaml_path}")
    return tuple(hp)


_DMS_EPS = 1e-8


def _dms_with_winner(
    box_rgb: torch.Tensor, var_rgb: torch.Tensor,
    box_ir:  torch.Tensor, var_ir:  torch.Tensor,
) -> tuple:
    """Wrapper around dms() that also returns the winning modality label.

    Returns
    -------
    best_box     : Tensor  — verbatim box from the lower-uncertainty stream
    sigma2_best  : Tensor  — scalar normalised variance of that stream
    winner       : str     — 'rgb' or 'ir'  (RGB wins ties, matching DMS)

    The winner label lets callers evaluate the prediction against the *correct*
    modality's GT without any cross-coordinate-space comparison.
    """
    # Replicate DMS's internal normalisation (scale_rgb/scale_ir) to infer
    # which branch dms() will pick — avoids floating-point allclose hacks.
    scale_rgb = torch.stack(
        [box_rgb[2], box_rgb[3], box_rgb[2], box_rgb[3]]
    ).clamp(min=_DMS_EPS)
    scale_ir = torch.stack(
        [box_ir[2], box_ir[3], box_ir[2], box_ir[3]]
    ).clamp(min=_DMS_EPS)
    s_rgb = (var_rgb / scale_rgb).mean().item()
    s_ir  = (var_ir  / scale_ir ).mean().item()
    winner = "rgb" if s_rgb <= s_ir else "ir"   # RGB wins ties — same as dms()
    best_box, sigma2_best = dms(box_rgb, var_rgb, box_ir, var_ir)
    return best_box, sigma2_best, winner


# ===========================================================================
# Configuration container
# ===========================================================================

@dataclass
class FusionConfig:
    """All knobs for one Phase 6.1 evaluation run.

    Threshold semantics (Phase 6.1, fixed for sanity check)
    -------------------------------------------------------
    theta_cls_detect : upper bound on u_fused for INIT (DETECT->TRACK)
    theta_loc_detect : upper bound on sigma2_best for INIT
    theta_cls_track  : upper bound on u_fused for MAINTAIN (stay in TRACK)
    theta_loc_track  : upper bound on sigma2_best for MAINTAIN

    The "track" thresholds are typically looser than the "detect" thresholds
    because once locked on, the tracker should be allowed to ride through
    short uncertainty spikes before reverting to redetection.
    """

    # Detection (YOLO) checkpoints + base weights for arch reconstruction
    yolo_ir_weights:  str = ""
    yolo_rgb_weights: str = ""
    yolo_ir_base:     str = ""    # original YOLOv5 .pt (architecture)
    yolo_rgb_base:    str = ""

    # Tracking (SiamCAR Stage 2) checkpoints + Stage 1 paths for arch
    siam_ir_stage2:    str = ""
    siam_rgb_stage2:   str = ""
    siam_ir_stage1:    str = ""
    siam_rgb_stage1:   str = ""
    siam_ir_pretrained:  str = ""   # original SiamCAR .pth used for arch init
    siam_rgb_pretrained: str = ""

    # SiamCAR YAML configs (one per modality — stride/SCORE_SIZE/ANTIUAVTHERMAL/RGB)
    siam_ir_cfg:  str = ""
    siam_rgb_cfg: str = ""

    # Fusion mode: "dbf" or "abf"
    fusion: str = "dbf"
    lam:    float = 1.0          # discount exponent (used only by DBF)
    K:      int = 2

    # YOLO uncertainty aggregation for _yolo_run:
    #   "mean" — average b/u/sigma2 over ALL pre-NMS anchors (legacy default;
    #            diluted by background anchors, but what the tracker thresholds
    #            were calibrated on — kept as default to preserve those results)
    #   "peak" — belief/uncertainty of the SELECTED detection only (the
    #            max-confidence anchor = top NMS survivor).  Consistent with
    #            SiamCAR's peak-sampled tracking uncertainty.
    unc_agg: str = "mean"

    # Belief assigned to a YOLO stream whose NMS returns nothing (DETECT phase):
    #   "dogmatic_bg" — legacy _ZERO_UNC placeholder b=(0,1), u=1 (used by all
    #                   original det+track runs).  An INVALID subjective-logic
    #                   opinion (b.sum() + u = 2) that lets a missing stream
    #                   "vote" background in the fusion.
    #   "vacuous"     — partition-consistent zero-evidence opinion b=(0,0), u=1,
    #                   matching what compute_beliefs yields for alpha=(1,1) and
    #                   the corrected det-only main results.
    miss_belief: str = "dogmatic_bg"

    # Switching thresholds (empirically calibrated from smoke-test distributions)
    #
    # Observed on sequence 20190925_101846_1_4 (1000 frames, UAV always present):
    #   u_fused     : 0.640 – 0.729  (global-mean over all YOLO anchors; structurally
    #                                  high because background cells dominate → floor ≈ 0.65)
    #   sigma2_best : 0.035 – 0.055  (DMS-normalised spatial variance)
    #
    # theta_cls_detect = 0.80 : covers YOLO u_fused max (0.73) + margin;
    #                            fails gracefully when neither stream detects (u = 1.0)
    # theta_loc_detect = 0.10 : gates on YOLO sigma2_best (observed 0.03–0.05); snug but passes
    # theta_cls_track  = 0.90 : gates on SiamCAR peak u (observed 0.16–0.61); SiamCAR u is lower
    #                            than YOLO global-mean u because it samples at the tracked peak
    # theta_loc_track  = 0.70 : gates on SiamCAR sigma2_best (empirically calibrated on smoke seq).
    #                            Good-tracking frames have sigma2 ≈ 0.46–0.68; the drift event
    #                            first exceeds 0.70 (observed: 0.722 on frame-5 of the smoke
    #                            sequence, exactly when IR IoU drops to 0.000).  Setting 0.70
    #                            catches the drift and triggers YOLO re-detection on the next
    #                            frame.  Previous value of 2.00 was inert — the drifted tracker
    #                            "ghost-tracks" background confidently with sigma2 ≈ 0.5–0.9.
    #
    # Important: theta_loc_detect and theta_loc_track operate on DIFFERENT signal distributions.
    # 0.10 is correct for the YOLO detection phase; 0.70 is correct for the SiamCAR tracking phase.
    #
    # Phase 6.2 will grid-search all four on the full val split.
    theta_cls_detect: float = 0.80
    theta_loc_detect: float = 0.10
    theta_cls_track:  float = 0.90
    theta_loc_track:  float = 0.70

    # YOLO inference knobs
    yolo_imgsz:        tuple = (640, 640)
    yolo_conf_thres:   float = 0.25
    yolo_iou_thres:    float = 0.45
    yolo_max_det:      int   = 300
    # Class filter for NMS — MUST be [0] for EvidentialYOLO with K=2.
    # Class 0 = UAV, class 1 = background.  classes=None would let
    # high-confidence "background" detections (class 1) win the NMS sort and
    # get passed to SiamCAR as the init box, causing it to track a background
    # patch for the entire sequence (IoU = 0 vs GT throughout).
    yolo_classes:      list  = None   # set in __post_init__ below

    # SiamCAR HP — overridden per-modality from the YAML's HP_SEARCH block
    hp_ir:  tuple | None = None   # (lr, penalty_k, window_lr)
    hp_rgb: tuple | None = None

    # Misc
    device: str = "cuda:0"

    def __post_init__(self):
        # Mutable default: yolo_classes must not be a shared list at class level.
        # Default = [0] = UAV class only.  Pass [] to disable class filtering.
        if self.yolo_classes is None:
            self.yolo_classes = [0]


# ===========================================================================
# Utility: extract YOLO uncertainty at the highest-confidence anchor
# ===========================================================================

_ZERO_UNC = {
    "b":           torch.tensor([0.0, 1.0]),        # all bg, dogmatic
    "u":           torch.tensor(1.0),               # vacuous — no information
    "sigma2_norm": torch.full((4,), 1e6),           # very large → DMS will NOT prefer
                                                    # a missing modality over a real one
}


def _yolo_run(image_bgr: np.ndarray,
              model: EvidentialYOLO,
              cfg: FusionConfig,
              device: torch.device,
              imgsz: tuple,
              stride: int):
    """Run one EvidentialYOLO model on one BGR frame.

    ``imgsz`` and ``stride`` must be precomputed once at tracker init time
    (calling check_img_size / model.stride.max() per frame is wasteful).
    ``model`` must already be in eval mode — do not call model.eval() here.

    Returns
    -------
    box_xywh : list[float] | None
        Top detection in [x, y, w, h] (image pixel coords), or None.
    unc : dict
        Keys: 'b' (Tensor[K]), 'u' (Tensor[]), 'sigma2_norm' (Tensor[4]).
        Values are the mean across ALL anchors at all feature-map scales
        (pre-NMS).  This dilutes the signal with background anchors but is
        consistent across all variants (DBF, ABF, single, gated), so
        comparative conclusions between variants remain valid.  Obtaining
        per-anchor uncertainty for only NMS-surviving detections would require
        matching det[0] back through unc_scales by index.
    """
    # BGR -> letterboxed RGB tensor on device
    im = letterbox(image_bgr, imgsz, stride=stride, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]      # HWC BGR -> CHW RGB
    im = np.ascontiguousarray(im)
    im_t = torch.from_numpy(im).to(device).float() / 255.0
    if im_t.ndim == 3:
        im_t = im_t.unsqueeze(0)

    with torch.no_grad():
        out = model(im_t)

    # EvidentialYOLO inference returns (nms_preds, base, var, evi, unc_scales)
    if not isinstance(out, tuple) or len(out) != 5:
        return None, dict(_ZERO_UNC)
    nms_preds, _base, _var, _evi, unc_scales = out

    # Filter to UAV class (0) only — class 1 is background.
    # With classes=None the NMS would accept high-confidence "background"
    # detections and return them as the top detection, causing SiamCAR to be
    # initialized on the wrong target.
    nms_classes = cfg.yolo_classes if cfg.yolo_classes else None
    pred = non_max_suppression(
        nms_preds,
        cfg.yolo_conf_thres,
        cfg.yolo_iou_thres,
        classes=nms_classes,
        agnostic=False,
        max_det=cfg.yolo_max_det,
    )
    det = pred[0]
    if det is None or len(det) == 0:
        return None, dict(_ZERO_UNC)

    det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], image_bgr.shape).round()
    best = det[0, :4].cpu().numpy()
    box_xywh = [
        float(best[0]),
        float(best[1]),
        float(best[2] - best[0] + 1),
        float(best[3] - best[1] + 1),
    ]

    # Aggregate uncertainty across scales.
    # NOTE: We expose *raw* sigma2 (not sigma2_norm) so DMS can do its own
    # box-size normalisation downstream — otherwise we'd double-normalise.
    #
    # cfg.unc_agg selects the aggregation (see FusionConfig):
    #   "mean" — average over ALL pre-NMS anchors (legacy default)
    #   "peak" — the SELECTED detection's anchor only (max-confidence anchor,
    #            i.e. the top NMS survivor: argmax(obj * b_uav))
    if getattr(cfg, "unc_agg", "mean") == "peak":
        conf_all = (nms_preds[0, :, 4] * nms_preds[0, :, 5])  # obj * b_uav
        peak = int(conf_all.argmax().item())
        b_sel = u_sel = sig_sel = None
        off = 0
        for s in unc_scales:
            _, na, ny, nx, _ = s["b"].shape
            size = na * ny * nx
            if peak < off + size:
                loc = peak - off
                a   = loc // (ny * nx)
                rem = loc % (ny * nx)
                r   = rem // nx
                c   = rem % nx
                b_sel   = s["b"][0, a, r, c].cpu()                  # (K,)
                u_sel   = s["u"][0, a, r, c].reshape(()).cpu()      # ()
                sig_sel = s["sigma2"][0, a, r, c].cpu()             # (4,) raw
                break
            off += size
        if b_sel is not None:
            return box_xywh, {"b": b_sel, "u": u_sel, "sigma2_norm": sig_sel}
        # Fall through to mean if the index map failed (should not happen).

    bs, us, sigs = [], [], []
    for s in unc_scales:
        bs.append(s["b"].reshape(-1, s["b"].shape[-1]))
        us.append(s["u"].reshape(-1))
        sigs.append(s["sigma2"].reshape(-1, s["sigma2"].shape[-1]))

    b_all   = torch.cat(bs, dim=0).mean(dim=0).cpu()       # (K,)
    u_all   = torch.cat(us).mean().cpu()                    # ()
    sig_all = torch.cat(sigs, dim=0).mean(dim=0).cpu()     # (4,) raw

    unc = {"b": b_all, "u": u_all, "sigma2_norm": sig_all}
    return box_xywh, unc


# ===========================================================================
# Utility: SiamCAR Stage-2 evidential tracker with peak-extracted uncertainty
# ===========================================================================

class _EvidentialSiamCARTracker(SiamCARTracker):
    """Reuses the localization logic of SiamCARTracker but reaches into the
    SiamCARStage2 model output to expose Dirichlet evidence and Gaussian
    log-variance at the predicted peak.

    Notes
    -----
    * The Stage-2 ``model.track()`` returns ``cls``, ``loc``, ``cen``,
      ``log_var`` and ``evi_raw`` of shape (1, *, 25, 25).  We use the
      Stage-1 ``cls`` output for localization (unchanged from the original
      SiamCAR pipeline) so the tracker behaviour matches the pretrained
      tracking quality, then sample evi_raw and log_var at the *upsampled*
      peak mapped back to the SCORE_SIZE grid.
    * The exemplar feature buffer ``self.zf`` is owned by the model.
    """

    def track(self, img_bgr, hp):
        """One step of tracking.  Returns dict with bbox + uncertainty.

        Parameters
        ----------
        img_bgr : np.ndarray HxWx3 BGR
        hp      : {'lr', 'penalty_k', 'window_lr'}

        Returns
        -------
        out : dict
            'bbox'        : [x, y, w, h] in image coords
            'b'           : Tensor(K,) belief masses at peak (UAV index 0)
            'u'           : Tensor()    vacuity at peak
            'sigma2_norm' : Tensor(4,)  bbox-size-normalised variances at peak
            'cls_score'   : float       softmax UAV prob at peak (debug)
        """
        w_z = self.size[0] + siam_cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + siam_cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = siam_cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (siam_cfg.TRACK.INSTANCE_SIZE / siam_cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(
            img_bgr, self.center_pos,
            siam_cfg.TRACK.INSTANCE_SIZE, round(s_x), self.channel_average,
        )

        outputs = self.model.track(x_crop)
        # Stage-2 outputs include evi_raw (1, K, H, W) and log_var (1, 4, H, W).
        cls_t = outputs["cls"]                # (1, 2, H, W) — Stage-1 logits
        evi_t = outputs.get("evi_raw")        # (1, K, H, W) or None
        var_t = outputs.get("log_var")        # (1, 4, H, W) or None

        cls_np = self._convert_cls(cls_t).squeeze()
        cen = outputs["cen"].data.cpu().numpy()
        cen = (cen - cen.min()) / np.ptp(cen)
        cen = cen.squeeze()
        lrtbs_np = outputs["loc"].data.cpu().numpy().squeeze()

        upsize = (siam_cfg.TRACK.SCORE_SIZE - 1) * siam_cfg.TRACK.STRIDE + 1
        penalty = self.cal_penalty(lrtbs_np, hp["penalty_k"])
        p_score = penalty * cls_np * cen
        if siam_cfg.TRACK.hanming:
            hp_score = p_score * (1 - hp["window_lr"]) + self.window * hp["window_lr"]
        else:
            hp_score = p_score

        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        p_score_up  = cv2.resize(p_score,  (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up      = cv2.resize(cls_np,   (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs       = np.transpose(lrtbs_np, (1, 2, 0))
        lrtbs_up    = cv2.resize(lrtbs,    (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / siam_cfg.TRACK.SCORE_SIZE
        max_r_up, max_c_up, new_cx, new_cy = self.getCenter(
            hp_score_up, p_score_up, scale_score, lrtbs,
        )

        # Map upsampled peak back to SCORE_SIZE grid for evi_raw / log_var
        max_r = int(np.clip(round(max_r_up / scale_score), 0, siam_cfg.TRACK.SCORE_SIZE - 1))
        max_c = int(np.clip(round(max_c_up / scale_score), 0, siam_cfg.TRACK.SCORE_SIZE - 1))

        # Box from upsampled lrtbs (unchanged SiamCAR behaviour)
        ave_w = (lrtbs_up[max_r_up, max_c_up, 0] + lrtbs_up[max_r_up, max_c_up, 2]) / self.scale_z
        ave_h = (lrtbs_up[max_r_up, max_c_up, 1] + lrtbs_up[max_r_up, max_c_up, 3]) / self.scale_z

        s_c = self.change(self.sz(ave_w, ave_h) /
                          self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty_w = np.exp(-(r_c * s_c - 1) * hp["penalty_k"])
        lr = penalty_w * cls_up[max_r_up, max_c_up] * hp["lr"]
        new_w = lr * ave_w + (1 - lr) * self.size[0]
        new_h = lr * ave_h + (1 - lr) * self.size[1]

        from pysot.utils.misc import bbox_clip
        cx     = bbox_clip(new_cx, 0, img_bgr.shape[1])
        cy     = bbox_clip(new_cy, 0, img_bgr.shape[0])
        width  = bbox_clip(new_w,  0, img_bgr.shape[1])
        height = bbox_clip(new_h,  0, img_bgr.shape[0])

        self.center_pos = np.array([cx, cy])
        self.size       = np.array([width, height])
        bbox = [cx - width / 2, cy - height / 2, width, height]

        # ── Extract uncertainty at the (max_r, max_c) location ─────────
        if evi_t is not None and var_t is not None:
            evidence = F.relu(evi_t[0, :, max_r, max_c])    # (K,)
            alpha = evidence + 1.0
            S = alpha.sum()
            b = (evidence / S).cpu()
            u = (evi_t.shape[1] / S).cpu()                  # K / S

            log_var = var_t[0, :, max_r, max_c].clamp(-10, 10)
            sigma2 = torch.exp(log_var).cpu()               # (4,) in ltrb space
            # SiamCAR's loc head outputs ltrb at search-grid scale.  For DMS we
            # only need a per-coord variance to be normalised by box dims, so
            # we reuse sigma2 directly (l, t, r, b ~ horiz, vert distances).
            # We map (l, t, r, b) variance -> (cx, cy, w, h) variance:
            #   var(cx) ~ (var(l) + var(r)) / 4
            #   var(cy) ~ (var(t) + var(b)) / 4
            #   var(w)  ~  var(l) + var(r)
            #   var(h)  ~  var(t) + var(b)
            v_l, v_t, v_r, v_b = sigma2.unbind()
            var_cx = (v_l + v_r) / 4.0
            var_cy = (v_t + v_b) / 4.0
            var_w  =  v_l + v_r
            var_h  =  v_t + v_b
            sigma2_box = torch.stack([var_cx, var_cy, var_w, var_h])
        else:
            b = torch.tensor([0.0, 1.0])
            u = torch.tensor(1.0)
            sigma2_box = torch.zeros(4)

        return {
            "bbox":        bbox,
            "b":           b,
            "u":           u,
            "sigma2_norm": sigma2_box,
            "cls_score":   float(cls_up[max_r_up, max_c_up]),
        }


# ===========================================================================
# Main fusion tracker
# ===========================================================================

class EDTCFusionTracker:
    """Dual-stream uncertainty-aware UAV detector + tracker.

    Two states:
        DETECT — both YOLO heads run; on success, both SiamCAR templates init
                 and we transition to TRACK.
        TRACK  — both SiamCAR heads run; on uncertainty violation we fall back
                 to DETECT.
    """

    DETECT = 0
    TRACK  = 1

    def __init__(self, config: FusionConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.state = self.DETECT

        # ── Load YOLO evidential heads ────────────────────────────────
        self.yolo_ir = EvidentialYOLO.load(
            config.yolo_ir_weights,  config.yolo_ir_base,  self.device, K=config.K,
        )
        self.yolo_rgb = EvidentialYOLO.load(
            config.yolo_rgb_weights, config.yolo_rgb_base, self.device, K=config.K,
        )
        self.yolo_ir.eval()
        self.yolo_rgb.eval()

        # Precompute once — check_img_size / stride.max() are constant per model.
        self._yolo_ir_stride  = int(self.yolo_ir.stride.max())
        self._yolo_rgb_stride = int(self.yolo_rgb.stride.max())
        self._yolo_ir_imgsz   = check_img_size(config.yolo_imgsz, s=self._yolo_ir_stride)
        self._yolo_rgb_imgsz  = check_img_size(config.yolo_imgsz, s=self._yolo_rgb_stride)

        # ── Load SiamCAR Stage-2 models ──────────────────────────────
        # SiamCAR uses a global YACS cfg.  IR and RGB configs share TRACK
        # params (SCORE_SIZE, STRIDE, EXEMPLAR_SIZE, ...) and only differ in
        # HP_SEARCH names (ANTIUAVTHERMAL vs ANTIUAVRGB) and dataset paths.
        # Merge IR for the active TRACK config; pull HP_SEARCH per modality
        # from the original YAMLs since the global cfg won't have both.
        siam_cfg.merge_from_file(config.siam_ir_cfg)

        self.siam_ir_model = SiamCARStage2.load(
            config.siam_ir_stage2,
            stage1_path=config.siam_ir_stage1,
            pretrained_path=config.siam_ir_pretrained,
            K=config.K,
        ).to(self.device).eval()

        self.siam_rgb_model = SiamCARStage2.load(
            config.siam_rgb_stage2,
            stage1_path=config.siam_rgb_stage1,
            pretrained_path=config.siam_rgb_pretrained,
            K=config.K,
        ).to(self.device).eval()

        self.siam_ir_tracker  = _EvidentialSiamCARTracker(self.siam_ir_model,  siam_cfg.TRACK)
        self.siam_rgb_tracker = _EvidentialSiamCARTracker(self.siam_rgb_model, siam_cfg.TRACK)

        self.hp_ir  = config.hp_ir  or _read_hp(config.siam_ir_cfg,  "ANTIUAVTHERMAL")
        self.hp_rgb = config.hp_rgb or _read_hp(config.siam_rgb_cfg, "ANTIUAVRGB")

        # ── Per-modality init flags ───────────────────────────────────
        # Tracks which SiamCAR trackers have been successfully initialised.
        # Each tracker is ONLY initialised with its OWN modality's YOLO box so
        # that pixel coordinates never cross the IR↔RGB boundary.
        self._ir_initialized  = False
        self._rgb_initialized = False

        # ── Telemetry — diagnostics for the sanity check ─────────────
        self.frame_log = []   # one dict per frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        self.state = self.DETECT
        self.frame_log = []
        self._ir_initialized  = False
        self._rgb_initialized = False

    def step(self, ir_bgr: np.ndarray, rgb_bgr: np.ndarray) -> dict:
        """Process one paired (IR, RGB) frame.  Returns prediction dict."""
        if self.state == self.DETECT:
            return self._detect_step(ir_bgr, rgb_bgr)
        return self._track_step(ir_bgr, rgb_bgr)

    # ------------------------------------------------------------------
    # State 1: DETECT
    # ------------------------------------------------------------------

    def _detect_step(self, ir_bgr, rgb_bgr) -> dict:
        ir_box,  ir_unc  = _yolo_run(ir_bgr,  self.yolo_ir,  self.cfg, self.device,
                                     self._yolo_ir_imgsz,  self._yolo_ir_stride)
        rgb_box, rgb_unc = _yolo_run(rgb_bgr, self.yolo_rgb, self.cfg, self.device,
                                     self._yolo_rgb_imgsz, self._yolo_rgb_stride)

        # On a miss, _yolo_run returns the _ZERO_UNC placeholder b=(0,1), u=1 —
        # a dogmatic background opinion that also claims full vacuity (violates
        # b.sum() + u = 1).  'vacuous' replaces it with the partition-consistent
        # zero-evidence opinion b=(0,0), u=1, matching the corrected det-only
        # main results.  Mirrors _SingleYOLO.__call__ in run_detector_only.py.
        if self.cfg.miss_belief == "vacuous":
            if ir_box is None:
                ir_unc = dict(ir_unc); ir_unc["b"] = torch.zeros_like(ir_unc["b"])
            if rgb_box is None:
                rgb_unc = dict(rgb_unc); rgb_unc["b"] = torch.zeros_like(rgb_unc["b"])

        # If neither stream detects: report zeros, stay in DETECT
        if ir_box is None and rgb_box is None:
            self._log("DETECT_NONE", None, None, None, None,
                      ir_bbox=None, rgb_bbox=None, dms_winner=None)
            return {"bbox": [0.0, 0.0, 0.0, 0.0], "state": "DETECT", "u_fused": 1.0,
                    "sigma2_best": float("inf"), "switched": False}

        # ── Belief fusion (uses ZERO_UNC for missing stream) ──────────
        # NOTE: we do NOT copy boxes across modalities here.  Each stream's
        # uncertainty is taken as-is: ZERO_UNC.u = 1.0 (vacuous) when the
        # stream has no detection, so the fused belief is dominated by the
        # detecting stream, and u_fused is naturally higher (harder to trigger
        # INIT when only one stream fires).
        b_fused, u_fused = self._fuse(rgb_unc["b"], rgb_unc["u"], ir_unc["b"], ir_unc["u"])

        # ── DMS: select winning modality, get sigma2_best scalar ─────────────
        # When both detect: _dms_with_winner() identifies the lower-σ² stream
        # and returns its box verbatim — *this* is the correct prediction
        # because the box is in the winner's own coordinate space, which matches
        # the GT we will evaluate it against (Option-B DMS-aware Acc).
        # When only one detects: winner is trivially that modality.
        if ir_box is not None and rgb_box is not None:
            best_box_t, sigma2_best_t, dms_winner = _dms_with_winner(
                torch.tensor(rgb_box, dtype=torch.float32), rgb_unc["sigma2_norm"],
                torch.tensor(ir_box,  dtype=torch.float32), ir_unc["sigma2_norm"],
            )
            out_box = best_box_t.tolist()
        elif ir_box is not None:
            sigma2_best_t = ir_unc["sigma2_norm"].mean()
            out_box       = ir_box
            dms_winner    = "ir"
        else:
            sigma2_best_t = rgb_unc["sigma2_norm"].mean()
            out_box       = rgb_box
            dms_winner    = "rgb"

        u_f     = float(u_fused.item())
        b_uav_f = float(b_fused[0].item())
        b_bg_f  = float(b_fused[1].item())
        s_f = float(sigma2_best_t.item() if torch.is_tensor(sigma2_best_t)
                    else sigma2_best_t)

        switched = False
        # Class check mirrors EDTC: argmax(b_fused)==UAV is a prerequisite.
        # b_uav > b_bg is the faithful K=2 argmax — stricter b_uav > 0.5 would
        # double-count uncertainty mass that the u_fused threshold handles.
        if b_uav_f > b_bg_f and u_f < self.cfg.theta_cls_detect and s_f < self.cfg.theta_loc_detect:
            # ── Modality-specific SiamCAR initialisation ─────────────
            # Each tracker is initialised with ITS OWN modality's detection
            # box so pixel coordinates never cross the IR↔RGB boundary.
            # Trackers for which YOLO found nothing are left uninitialised
            # (flagged False) and skipped during the tracking step.
            self._ir_initialized  = False
            self._rgb_initialized = False

            if ir_box is not None:
                self.siam_ir_tracker.init(ir_bgr,  ir_box)
                self._ir_initialized = True

            if rgb_box is not None:
                self.siam_rgb_tracker.init(rgb_bgr, rgb_box)
                self._rgb_initialized = True

            # At least one tracker must initialise to enter TRACK mode.
            if self._ir_initialized or self._rgb_initialized:
                self.state = self.TRACK
                switched = True

        self._log("DETECT", out_box, u_f, s_f, switched,
                  ir_bbox=ir_box, rgb_bbox=rgb_box, dms_winner=dms_winner, b_uav=b_uav_f)
        return {"bbox": out_box, "state": "DETECT", "u_fused": u_f,
                "b_uav": b_uav_f, "sigma2_best": s_f, "switched": switched}

    # ------------------------------------------------------------------
    # State 2: TRACK
    # ------------------------------------------------------------------

    def _track_step(self, ir_bgr, rgb_bgr) -> dict:
        hp_ir  = {"lr": self.hp_ir[0],  "penalty_k": self.hp_ir[1],  "window_lr": self.hp_ir[2]}
        hp_rgb = {"lr": self.hp_rgb[0], "penalty_k": self.hp_rgb[1], "window_lr": self.hp_rgb[2]}

        # ── Only run trackers that were successfully initialised ──────────────
        # Skipping an uninitialised tracker prevents SiamCAR from trying to
        # track from a zero/garbage template and returning nonsense coordinates.
        with torch.no_grad():
            ir_out  = self.siam_ir_tracker.track(ir_bgr,   hp_ir)  if self._ir_initialized  else None
            rgb_out = self.siam_rgb_tracker.track(rgb_bgr, hp_rgb) if self._rgb_initialized else None

        # ── Safety: both trackers dropped (should never happen in practice) ──
        if ir_out is None and rgb_out is None:
            self.state = self.DETECT
            self._log("TRACK", [0.0, 0.0, 0.0, 0.0], 1.0, float("inf"), True,
                      ir_bbox=None, rgb_bbox=None, dms_winner=None)
            return {"bbox": [0.0, 0.0, 0.0, 0.0], "state": "TRACK",
                    "u_fused": 1.0, "sigma2_best": float("inf"), "switched": True}

        # ── Belief fusion (whichever trackers are active) ─────────────────────
        if ir_out is not None and rgb_out is not None:
            b_fused, u_fused = self._fuse(rgb_out["b"], rgb_out["u"],
                                          ir_out["b"],  ir_out["u"])
        elif ir_out is not None:
            b_fused, u_fused = ir_out["b"], ir_out["u"]
        else:
            b_fused, u_fused = rgb_out["b"], rgb_out["u"]

        # ── DMS: select winning modality, compute sigma2_best for gating ──────
        # When both trackers run: _dms_with_winner() identifies the lower-σ²
        # stream and returns its bbox verbatim — this is the correct prediction
        # for Option-B DMS-aware Acc (each bbox stays in its own coord space).
        # When only one runs: trivially that modality wins.
        ir_bbox_log  = ir_out["bbox"]  if ir_out  is not None else None
        rgb_bbox_log = rgb_out["bbox"] if rgb_out is not None else None

        if ir_out is not None and rgb_out is not None:
            best_box_t, sigma2_best_t, dms_winner = _dms_with_winner(
                torch.tensor(rgb_out["bbox"], dtype=torch.float32), rgb_out["sigma2_norm"],
                torch.tensor(ir_out["bbox"],  dtype=torch.float32), ir_out["sigma2_norm"],
            )
            best_box = best_box_t.tolist()
        elif ir_out is not None:
            sigma2_best_t = ir_out["sigma2_norm"].mean()
            best_box      = ir_out["bbox"]
            dms_winner    = "ir"
        else:
            sigma2_best_t = rgb_out["sigma2_norm"].mean()
            best_box      = rgb_out["bbox"]
            dms_winner    = "rgb"

        u_f     = float(u_fused.item())
        b_uav_f = float(b_fused[0].item())
        b_bg_f  = float(b_fused[1].item())
        s_f = float(sigma2_best_t.item() if torch.is_tensor(sigma2_best_t)
                    else sigma2_best_t)

        # MAINTAIN check — failure of any condition reverts to DETECT.
        # Class check (b_uav > b_bg) mirrors EDTC's argmax==TARGET guard.
        keep_tracking = (b_uav_f > b_bg_f and u_f < self.cfg.theta_cls_track and s_f < self.cfg.theta_loc_track)
        if not keep_tracking:
            self.state = self.DETECT

        self._log("TRACK", best_box, u_f, s_f, not keep_tracking,
                  ir_bbox=ir_bbox_log, rgb_bbox=rgb_bbox_log, dms_winner=dms_winner, b_uav=b_uav_f)
        return {"bbox": best_box, "state": "TRACK", "u_fused": u_f,
                "b_uav": b_uav_f, "sigma2_best": s_f, "switched": (not keep_tracking)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fuse(self, b_rgb, u_rgb, b_ir, u_ir):
        if self.cfg.fusion == "abf":
            return abf(b_rgb, u_rgb, b_ir, u_ir)
        return dbf(b_rgb, u_rgb, b_ir, u_ir, lam=self.cfg.lam)

    def _log(self, mode, box, u, s, switched,
             ir_bbox=None, rgb_bbox=None, dms_winner=None, b_uav=None):
        """Append one frame's telemetry.

        ir_bbox / rgb_bbox : raw per-modality bboxes (for Option-B DMS eval)
        dms_winner         : 'ir', 'rgb', or None (for coord-space routing)
        b_uav              : b_fused(UAV) — fused belief mass for UAV class [0,1]
        """
        self.frame_log.append({
            "state":       mode,
            "bbox":        box,
            "u_fused":     u,
            "b_uav":       b_uav,
            "sigma2_best": s,
            "switched":    switched,
            "ir_bbox":     ir_bbox,
            "rgb_bbox":    rgb_bbox,
            "dms_winner":  dms_winner,
        })
