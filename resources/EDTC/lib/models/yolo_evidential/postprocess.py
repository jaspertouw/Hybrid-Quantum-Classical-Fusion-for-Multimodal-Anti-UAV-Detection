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
    if prediction.ndim != 2 or prediction.shape[1] != 6:
        raise ValueError(
            "prediction must have shape [anchors, 6] with one UAV class"
        )
    if not 0.0 <= score_minimum <= 1.0:
        raise ValueError("score_minimum must lie in [0, 1]")
    if not torch.isfinite(prediction).all():
        raise RuntimeError("non-finite detector prediction")

    objectness = prediction[:, 4]
    b_uav = prediction[:, 5]
    scores = objectness * b_uav
    eligible = torch.nonzero(scores >= score_minimum, as_tuple=False).flatten()
    if eligible.numel() == 0:
        return None

    eligible_scores = scores[eligible]
    best_score = eligible_scores.max()
    # nonzero preserves flattened anchor order, making exact ties deterministic.
    tied = eligible[eligible_scores == best_score]
    anchor_index = int(tied[0].item())

    flattened = {
        key: _flatten_scale_field(uncertainty_scales, key, batch_index)
        for key in REQUIRED_UNCERTAINTY_KEYS
    }
    anchor_count = prediction.shape[0]
    for key, values in flattened.items():
        if values.shape[0] != anchor_count:
            raise RuntimeError(
                f"{key} has {values.shape[0]} anchors; expected {anchor_count}"
            )
        if not torch.isfinite(values).all():
            raise RuntimeError(f"non-finite selected-output field: {key}")

    beliefs = flattened["b"][anchor_index]
    uncertainty = flattened["u"][anchor_index, 0]
    partition_error = (beliefs.sum() + uncertainty - 1.0).abs()
    if partition_error.item() > 1e-5:
        raise RuntimeError(
            f"belief partition mismatch at anchor {anchor_index}: "
            f"{partition_error.item():.3g}"
        )
    if (beliefs < 0).any() or not 0.0 < uncertainty.item() <= 1.0:
        raise RuntimeError("invalid evidential masses at selected anchor")
    if any((flattened[key][anchor_index] <= 0).any() for key in (
        "sigma2_grid", "sigma2_pixel"
    )):
        raise RuntimeError("selected anchor has non-positive coordinate variance")

    xywh = prediction[anchor_index, :4]
    xyxy = xywh2xyxy(xywh.unsqueeze(0))[0]
    return {
        "anchor_index": anchor_index,
        "box_xywh_input": xywh,
        "box_xyxy_input": xyxy,
        "candidate_score": scores[anchor_index],
        "objectness": objectness[anchor_index],
        "beliefs": beliefs,
        "uncertainty": uncertainty,
        "sigma2_grid": flattened["sigma2_grid"][anchor_index],
        "sigma2_pixel": flattened["sigma2_pixel"][anchor_index],
        "sigma2_norm_corrected": flattened["sigma2_norm_corrected"][anchor_index],
        "sigma2_norm_inherited_head": flattened["sigma2_norm_inherited"][anchor_index],
    }


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
