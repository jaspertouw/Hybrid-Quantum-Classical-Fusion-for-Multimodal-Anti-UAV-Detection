"""Exact association of an NMS survivor with its evidential anchor output."""

import torch

from yolov5.utils.general import xywh2xyxy


def selected_anchor_index(prediction, selected_detection):
    """Return the pre-NMS anchor index for a one-class selected detection."""
    if selected_detection.numel() == 0:
        raise ValueError("selected detection is empty")
    boxes = xywh2xyxy(prediction[:, :4])
    confidence = prediction[:, 4] * prediction[:, 5]
    target_box = selected_detection[:4]
    target_confidence = selected_detection[4]
    error = (boxes - target_box).abs().amax(dim=1) + (confidence - target_confidence).abs()
    index = int(error.argmin().item())
    if not torch.allclose(boxes[index], target_box, atol=1e-3, rtol=1e-5):
        raise RuntimeError("could not associate NMS survivor with a detector anchor")
    if not torch.allclose(confidence[index], target_confidence, atol=1e-6, rtol=1e-5):
        raise RuntimeError("selected-anchor confidence mismatch")
    return index


def selected_anchor_uncertainty(prediction, uncertainty_scales, selected_detection, batch_index=0):
    """Extract uncertainty from the exact anchor that produced the survivor."""
    index = selected_anchor_index(prediction, selected_detection)

    def flattened(key):
        values = [scale[key][batch_index].reshape(-1, scale[key].shape[-1])
                  for scale in uncertainty_scales]
        return torch.cat(values, dim=0)

    return {
        "anchor_index": index,
        "beliefs": flattened("b")[index],
        "uncertainty": flattened("u")[index, 0],
        "sigma2_grid": flattened("sigma2_grid")[index],
        "sigma2_pixel": flattened("sigma2_pixel")[index],
        "sigma2_norm_corrected": flattened("sigma2_norm_corrected")[index],
        "sigma2_norm_inherited": flattened("sigma2_norm_inherited")[index],
    }
