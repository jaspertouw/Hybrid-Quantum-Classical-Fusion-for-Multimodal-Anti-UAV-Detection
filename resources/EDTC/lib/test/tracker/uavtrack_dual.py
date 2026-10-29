"""
Dual-stream UAV tracker — Phase 3.1: Parallel YOLO detection heads.

Both IR and RGB streams each pass through their own independent YOLOv5 model.
No cross-modal fusion is performed at this stage.

Architecture:
  - IR YOLO  (best_clean.pt)    → detects UAV in thermal frames
  - RGB YOLO (best_clean_rgb.pt) → detects UAV in visible frames
  - Transformer backbone (UAVTrackEH) tracks using IR as primary modality,
    exactly as in the single-stream baseline.

Detection/re-detection events trigger both heads simultaneously so that
both streams always produce an up-to-date bounding box estimate.
Between re-detection events the last known RGB box is carried forward.

Output keys per frame:
  target_bbox      — IR tracker box (same as ir_bbox; kept for eval compat)
  ir_bbox          — bounding box from IR YOLO / IR transformer track
  rgb_bbox         — bounding box from RGB YOLO (updated on detect events)
  uav_score        — evidential target probability (IR transformer)
  uav_uncertainty  — epistemic uncertainty (IR transformer)
  detection_flag   — True while no valid target has been found in IR stream
"""

import torch
import numpy as np
import cv2
import os

from lib.test.tracker.uavtrack_eh import MixFormerOnline
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box, relu_evidence

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (
    check_img_size, non_max_suppression, scale_boxes,
)
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device


_ZERO_BOX = [0, 0, 0, 0]


class DualStreamTracker(MixFormerOnline):
    """
    Extends MixFormerOnline with a parallel RGB YOLO detection head.
    The transformer backbone is unchanged and still operates on IR frames.
    """

    # ------------------------------------------------------------------
    # YOLO loading
    # ------------------------------------------------------------------

    def initialize_yolo_dual(self):
        """Load both IR and RGB YOLO models and store their devices.

        Call this once before the tracking loop instead of initialize_yolo().
        Returns (yolo_ir, yolo_rgb).
        """
        ir_cfg = self.cfg.YOLO_IR
        rgb_cfg = self.cfg.YOLO_RGB

        self._ir_device = select_device(ir_cfg.DEVICE)
        self.yolo_ir = DetectMultiBackend(
            ir_cfg.WEIGHTS, device=self._ir_device,
            dnn=False, data=ir_cfg.DATA, fp16=False,
        )

        self._rgb_device = select_device(rgb_cfg.DEVICE)
        self.yolo_rgb = DetectMultiBackend(
            rgb_cfg.WEIGHTS, device=self._rgb_device,
            dnn=False, data=rgb_cfg.DATA, fp16=False,
        )

        # Keep self.yolo_model pointing to IR for any parent-class code paths
        self.yolo_model = self.yolo_ir

        # Initialise last known RGB box so track() is safe before first detect
        self.last_rgb_box = list(_ZERO_BOX)

        return self.yolo_ir, self.yolo_rgb

    # ------------------------------------------------------------------
    # Internal helper: run one YOLO model on one image
    # ------------------------------------------------------------------

    def _run_yolo(self, image, model, yolo_cfg, device):
        """Run a YOLOv5 model on *image* and return the top detection box.

        Args:
            image:    HxWx3 numpy array (RGB, uint8).
            model:    DetectMultiBackend instance.
            yolo_cfg: edict with IMGSZ, CONF_THRES, IOU_THRES, MAX_DET,
                      AUGMENT, CLASSES, AGNOSTIC_NMS fields.
            device:   torch.device on which the model lives.

        Returns:
            box_xywh (list[float] | None): [x, y, w, h] of best detection,
                or None when nothing is detected.
        """
        imgsz = check_img_size(yolo_cfg.IMGSZ, s=model.stride)

        # Pre-process
        im = letterbox(image, imgsz, stride=model.stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]          # HWC BGR→CHW RGB
        im = np.ascontiguousarray(im)

        model.warmup(imgsz=(1, 3, *imgsz))

        im_t = torch.from_numpy(im).to(device)
        im_t = im_t.half() if model.fp16 else im_t.float()
        im_t /= 255.0
        if im_t.ndim == 3:
            im_t = im_t.unsqueeze(0)

        # Forward + NMS
        pred = model(im_t, augment=yolo_cfg.AUGMENT, visualize=False)
        pred = non_max_suppression(
            pred,
            yolo_cfg.CONF_THRES,
            yolo_cfg.IOU_THRES,
            yolo_cfg.CLASSES,
            yolo_cfg.AGNOSTIC_NMS,
            max_det=yolo_cfg.MAX_DET,
        )

        det = pred[0]
        if len(det) == 0:
            return None

        # Scale boxes back to original image size
        det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], image.shape).round()

        # Return highest-confidence detection in [x, y, w, h] format
        best = det[0, :4].cpu().numpy()
        box_xywh = [
            float(best[0]),
            float(best[1]),
            float(best[2] - best[0] + 1),
            float(best[3] - best[1] + 1),
        ]
        return box_xywh

    # ------------------------------------------------------------------
    # Initialize: detect on both streams, init transformer with IR box
    # ------------------------------------------------------------------

    def initialize(self, ir_image, rgb_image, ir_model, rgb_model):
        """Detect the UAV in both streams and initialise the IR transformer.

        Args:
            ir_image:  HxWx3 numpy array, thermal frame (RGB channel order).
            rgb_image: HxWx3 numpy array, visible frame (RGB channel order).
            ir_model:  DetectMultiBackend for IR.
            rgb_model: DetectMultiBackend for RGB.

        Returns:
            dict with keys: target_bbox, ir_bbox, rgb_bbox, uav_score,
                            detection_flag.
        """
        ir_box = self._run_yolo(ir_image, ir_model, self.cfg.YOLO_IR, self._ir_device)
        rgb_box = self._run_yolo(rgb_image, rgb_model, self.cfg.YOLO_RGB, self._rgb_device)

        # Always update the stored RGB box so track() can return it
        self.last_rgb_box = rgb_box if rgb_box is not None else list(_ZERO_BOX)

        if ir_box is None:
            # No IR detection — remain in detection mode
            return {
                'target_bbox': list(_ZERO_BOX),
                'ir_bbox': list(_ZERO_BOX),
                'rgb_bbox': self.last_rgb_box,
                'uav_score': 0,
                'detection_flag': True,
            }

        # Initialise the transformer tracker with the IR detection
        self.inilialize_tracker(ir_image, ir_box)

        return {
            'target_bbox': ir_box,
            'ir_bbox': ir_box,
            'rgb_bbox': self.last_rgb_box,
            'uav_score': 1,
            'detection_flag': False,
        }

    # ------------------------------------------------------------------
    # Track: IR transformer forward; re-detect with both YOLOs on loss
    # ------------------------------------------------------------------

    def track(self, ir_image, rgb_image, info=None):
        """Track with the IR transformer; re-detect on both streams when uncertain.

        Args:
            ir_image:  HxWx3 numpy array, current thermal frame.
            rgb_image: HxWx3 numpy array, current visible frame.
            info:      optional metadata dict (unused, kept for API compat).

        Returns:
            dict with keys: target_bbox, ir_bbox, rgb_bbox, uav_score,
                            uav_uncertainty, detection_flag.
        """
        H, W, _ = ir_image.shape
        self.frame_id += 1

        x_patch_arr, resize_factor, _ = sample_target(
            ir_image, self.state, self.params.search_factor,
            output_sz=self.params.search_size,
        )
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            out_dict, _ = self.network(
                self.template, self.online_template, search, run_score_head=True,
            )

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        evidence = relu_evidence(out_dict['pred_scores'].squeeze(1))
        alpha = evidence + 1
        uncertainty = 2 / torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        _, preds = torch.max(out_dict['pred_scores'].squeeze(1), 1)

        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=5)

        # Re-detection: transformer is uncertain or predicts background
        if preds == 0 or (preds == 1 and uncertainty > self.params.evidential_threshold):
            out = self.initialize(ir_image, rgb_image, self.yolo_ir, self.yolo_rgb)
            return out

        # Successful track — RGB box is the last one from a detect event
        return {
            'target_bbox': self.state,
            'ir_bbox': self.state,
            'rgb_bbox': self.last_rgb_box,
            'uav_score': prob[0][1].item(),
            'uav_uncertainty': uncertainty[0][0].item(),
            'detection_flag': False,
        }

    # ------------------------------------------------------------------
    # Debug visualisation (optional)
    # ------------------------------------------------------------------

    def _debug_draw(self, ir_image, rgb_image, ir_box, rgb_box):
        """Draw both detection boxes and save side-by-side debug image."""
        def draw_box(img, box, color):
            if box and box[2] > 0:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        ir_vis = cv2.cvtColor(ir_image, cv2.COLOR_RGB2BGR)
        rgb_vis = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        draw_box(ir_vis, ir_box, (0, 0, 255))    # red  for IR
        draw_box(rgb_vis, rgb_box, (0, 255, 0))  # green for RGB

        # Pad to same height before hstack
        h_ir, w_ir = ir_vis.shape[:2]
        h_rgb, w_rgb = rgb_vis.shape[:2]
        if h_ir != h_rgb:
            max_h = max(h_ir, h_rgb)
            ir_vis = cv2.copyMakeBorder(ir_vis, 0, max_h - h_ir, 0, 0, cv2.BORDER_CONSTANT)
            rgb_vis = cv2.copyMakeBorder(rgb_vis, 0, max_h - h_rgb, 0, 0, cv2.BORDER_CONSTANT)

        combined = np.hstack([ir_vis, rgb_vis])
        save_path = os.path.join(self.save_dir, "dual_%04d.jpg" % self.frame_id)
        cv2.imwrite(save_path, combined)


def get_tracker_class():
    return DualStreamTracker
