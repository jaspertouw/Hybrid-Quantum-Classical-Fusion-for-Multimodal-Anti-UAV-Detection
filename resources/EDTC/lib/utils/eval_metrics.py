"""
Standalone evaluation metric utilities for UAV tracking experiments.

All functions are pure (no I/O, no side effects) and operate on numpy arrays.
Expected inputs per sequence are described in each function's docstring.

Confidence signals:
  - conf_prob: prob[0][1] = Dirichlet expected probability of target presence (range [0, 1])
  - conf_1mu:  1 - uncertainty = 1 - (2 / sum(alpha))                        (range [0, 1])

Correctness labels (binary, per frame, over full sequence including absent frames):
  - exist=True  frames: correct = iou(pred, gt) >= iou_threshold
  - exist=False frames: correct = model predicts background (conf_prob < 0.5)
"""

import numpy as np


def iou_xywh(bbox1, bbox2):
    """IoU between two boxes in [x, y, w, h] format."""
    bbox1 = [float(v) for v in bbox1]
    bbox2 = [float(v) for v in bbox2]
    x0_1, y0_1, w1, h1 = bbox1
    x0_2, y0_2, w2, h2 = bbox2
    x1_1, y1_1 = x0_1 + w1, y0_1 + h1
    x1_2, y1_2 = x0_2 + w2, y0_2 + h2
    ox0, oy0 = max(x0_1, x0_2), max(y0_1, y0_2)
    ox1, oy1 = min(x1_1, x1_2), min(y1_1, y1_2)
    if ox1 - ox0 <= 0 or oy1 - oy0 <= 0:
        return 0.0
    intersection = (ox1 - ox0) * (oy1 - oy0)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def compute_accuracy(correctness):
    """
    Mean accuracy over all frames.

    Args:
        correctness: (N,) bool/int array — 1 if prediction correct, 0 otherwise.

    Returns:
        float in [0, 1].
    """
    correctness = np.asarray(correctness, dtype=float)
    return float(np.mean(correctness))


def compute_mean_iou(ious):
    """
    Mean IoU over present frames (exist=True).

    Args:
        ious: (N,) float array of per-frame IoU values (only present frames).

    Returns:
        float in [0, 1].
    """
    ious = np.asarray(ious, dtype=float)
    return float(np.mean(ious)) if len(ious) > 0 else 0.0


def compute_ece(confs, correctness, n_bins=10):
    """
    Expected Calibration Error (ECE) over all frames.

    Args:
        confs:       (N,) float array of confidence scores in [0, 1].
        correctness: (N,) bool/int array — 1 if prediction correct, 0 otherwise.
        n_bins:      number of equal-width bins (default 10).

    Returns:
        float >= 0. Lower is better.
    """
    confs = np.asarray(confs, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    n = len(confs)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs >= bins[i]) & (confs < bins[i + 1])
        if i == n_bins - 1:
            mask = (confs >= bins[i]) & (confs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc_bin = correctness[mask].mean()
        conf_bin = confs[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_auroc(confs, correctness):
    """
    Area Under the ROC Curve (AUROC) over all frames.

    Requires at least one positive (correct) and one negative (incorrect) frame.

    Args:
        confs:       (N,) float array of confidence scores in [0, 1].
        correctness: (N,) bool/int array — 1 if prediction correct, 0 otherwise.

    Returns:
        float in [0, 1]. Higher is better. Returns NaN if only one class present.
    """
    from sklearn.metrics import roc_auc_score
    confs = np.asarray(confs, dtype=float)
    correctness = np.asarray(correctness, dtype=int)
    if len(np.unique(correctness)) < 2:
        return float('nan')
    return float(roc_auc_score(correctness, confs))


def compute_sparsification(ious, confs, fractions=None):
    """
    Sparsification curve and AUSE (Area Under Sparsification Error).

    Frames are retained in order of decreasing confidence. The curve shows how
    mean IoU improves as low-confidence (uncertain) predictions are removed.
    Restricted to present frames (exist=True) since IoU is undefined for absent frames.

    Args:
        ious:      (N,) float array of per-frame IoU values (present frames only).
        confs:     (N,) float array of confidence scores for those same frames.
        fractions: (M,) float array of retention fractions to evaluate (default: linspace).

    Returns:
        dict with keys:
          'fractions':    (M,) retention fractions
          'tracker_iou':  (M,) mean IoU of tracker at each retention fraction
          'oracle_iou':   (M,) mean IoU of oracle (sorted by actual IoU, best-first removed)
          'ause':         scalar AUSE (lower is better)
    """
    ious = np.asarray(ious, dtype=float)
    confs = np.asarray(confs, dtype=float)
    n = len(ious)
    if n == 0:
        return {'fractions': np.array([]), 'tracker_iou': np.array([]),
                'oracle_iou': np.array([]), 'ause': float('nan')}

    if fractions is None:
        fractions = np.linspace(0.1, 1.0, 19)
    fractions = np.asarray(fractions, dtype=float)

    # Sort by confidence descending: highest confidence retained first
    conf_order = np.argsort(confs)[::-1]
    # Oracle: sort by IoU descending (worst predictions removed first)
    oracle_order = np.argsort(ious)[::-1]

    tracker_iou = np.zeros(len(fractions))
    oracle_iou = np.zeros(len(fractions))

    for i, frac in enumerate(fractions):
        k = max(1, int(np.ceil(frac * n)))
        tracker_iou[i] = ious[conf_order[:k]].mean()
        oracle_iou[i] = ious[oracle_order[:k]].mean()

    # AUSE: area between oracle and tracker curves (oracle is always >= tracker)
    ause = float(np.trapz(oracle_iou - tracker_iou, fractions))

    return {
        'fractions': fractions,
        'tracker_iou': tracker_iou,
        'oracle_iou': oracle_iou,
        'ause': ause,
    }


def compute_fps(times):
    """
    Mean FPS from per-frame processing times.

    Args:
        times: (N,) float array of per-frame times in seconds.

    Returns:
        float FPS. Returns NaN if times is empty or mean is zero.
    """
    times = np.asarray(times, dtype=float)
    times = times[times > 0]
    if len(times) == 0:
        return float('nan')
    return float(1.0 / np.mean(times))
