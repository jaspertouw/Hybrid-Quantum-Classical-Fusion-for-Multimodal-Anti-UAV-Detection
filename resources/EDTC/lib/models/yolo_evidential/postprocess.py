"""Selected-anchor extraction for evidential YOLO outputs.

The successor path selects directly from the pre-NMS tensor.  For the frozen
one-class, non-merge NMS configuration, its highest-scoring eligible anchor is
necessarily the highest-scoring survivor: NMS processes that anchor first, so
no other anchor can suppress it.  Direct selection therefore preserves exact
anchor identity and avoids the ambiguity of matching a detection back by its
box and confidence.

The older matching helpers remain for reproduction checks only.
"""

import torch

from yolov5.utils.general import xywh2xyxy


REQUIRED_UNCERTAINTY_KEYS = (
    "b",
    "u",
    "sigma2_grid",
    "sigma2_pixel",
    "sigma2_norm_corrected",
    "sigma2_norm_inherited",
)


def _flatten_scale_field(uncertainty_scales, key, batch_index):
    values = [
        scale[key][batch_index].reshape(-1, scale[key].shape[-1])
        for scale in uncertainty_scales
    ]
    return torch.cat(values, dim=0)


def _flatten_scale_field_batch(uncertainty_scales, key):
    values = [
        scale[key].reshape(scale[key].shape[0], -1, scale[key].shape[-1])
        for scale in uncertainty_scales
    ]
    return torch.cat(values, dim=1)


def select_successor_batch(predictions, uncertainty_scales, score_minimum=0.25):
    """Vectorized exact-anchor extraction for one inference batch."""
    if predictions.ndim != 3 or predictions.shape[2] != 6:
        raise ValueError("predictions must have shape [batch, anchors, 6]")
    if not 0.0 <= score_minimum <= 1.0:
        raise ValueError("score_minimum must lie in [0, 1]")
    if not torch.isfinite(predictions).all():
        raise RuntimeError("non-finite detector prediction")

    flattened = {
        key: _flatten_scale_field_batch(uncertainty_scales, key)
        for key in REQUIRED_UNCERTAINTY_KEYS
    }
    batch_size, anchor_count, _ = predictions.shape
    for key, values in flattened.items():
        if values.shape[:2] != (batch_size, anchor_count):
            raise RuntimeError(
                f"{key} has shape {tuple(values.shape[:2])}; "
                f"expected {(batch_size, anchor_count)}"
            )
        if not torch.isfinite(values).all():
            raise RuntimeError(f"non-finite selected-output field: {key}")

    objectness = predictions[:, :, 4]
    scores = objectness * predictions[:, :, 5]
    eligible_scores = scores.masked_fill(scores < score_minimum, -1.0)
    # torch.max returns the first occurrence, which freezes the lowest-index tie rule.
    best_scores, best_indices = eligible_scores.max(dim=1)
    results = []
    for batch_index in range(batch_size):
        if best_scores[batch_index].item() < 0.0:
            results.append(None)
            continue
        anchor_index = int(best_indices[batch_index].item())
        beliefs = flattened["b"][batch_index, anchor_index]
        uncertainty = flattened["u"][batch_index, anchor_index, 0]
        partition_error = (beliefs.sum() + uncertainty - 1.0).abs()
        if partition_error.item() > 1e-5:
            raise RuntimeError(
                f"belief partition mismatch at batch {batch_index}, "
                f"anchor {anchor_index}: {partition_error.item():.3g}"
            )
        if (beliefs < 0).any() or not 0.0 < uncertainty.item() <= 1.0:
            raise RuntimeError("invalid evidential masses at selected anchor")
        if any((flattened[key][batch_index, anchor_index] <= 0).any() for key in (
            "sigma2_grid", "sigma2_pixel"
        )):
            raise RuntimeError("selected anchor has non-positive coordinate variance")

        xywh = predictions[batch_index, anchor_index, :4]
        results.append({
            "anchor_index": anchor_index,
            "box_xywh_input": xywh,
            "box_xyxy_input": xywh2xyxy(xywh.unsqueeze(0))[0],
            "candidate_score": scores[batch_index, anchor_index],
            "objectness": objectness[batch_index, anchor_index],
            "beliefs": beliefs,
            "uncertainty": uncertainty,
            "sigma2_grid": flattened["sigma2_grid"][batch_index, anchor_index],
            "sigma2_pixel": flattened["sigma2_pixel"][batch_index, anchor_index],
            "sigma2_norm_corrected": flattened["sigma2_norm_corrected"][batch_index, anchor_index],
            "sigma2_norm_inherited_head": flattened["sigma2_norm_inherited"][batch_index, anchor_index],
        })
    return results


def select_successor_candidate(
    prediction,
    uncertainty_scales,
    batch_index=0,
    score_minimum=0.25,
):
    """Return the exact top UAV anchor under the successor inference contract.

    ``prediction`` is one image's ``[cx, cy, w, h, objectness, b_uav]``
    tensor.  The threshold is inclusive, as frozen in the thesis protocol.
    Ties are resolved by the lowest flattened anchor index.  ``None`` denotes
    a vacuous detector miss; missing-feature imputation happens later after the
    training-only spatial cap has been fitted.
    """
    if prediction.ndim != 2:
        raise ValueError("prediction must have shape [anchors, 6]")
    batch_scales = [
        {key: value[batch_index:batch_index + 1] for key, value in scale.items()}
        for scale in uncertainty_scales
    ]
    return select_successor_batch(
        prediction.unsqueeze(0), batch_scales, score_minimum=score_minimum
    )[0]


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
        return _flatten_scale_field(uncertainty_scales, key, batch_index)

    return {
        "anchor_index": index,
        "beliefs": flattened("b")[index],
        "uncertainty": flattened("u")[index, 0],
        "sigma2_grid": flattened("sigma2_grid")[index],
        "sigma2_pixel": flattened("sigma2_pixel")[index],
        "sigma2_norm_corrected": flattened("sigma2_norm_corrected")[index],
        "sigma2_norm_inherited": flattened("sigma2_norm_inherited")[index],
    }
