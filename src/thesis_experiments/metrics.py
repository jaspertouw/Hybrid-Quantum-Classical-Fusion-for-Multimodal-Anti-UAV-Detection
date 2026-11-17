"""Sensor-native evaluation shared by every successor fusion model."""
from __future__ import annotations

import numpy as np

from .paired_evaluation import Candidate, corrected_dms_winner, frame_iou


def evaluation_data(rows, sensor=None):
    if not rows or any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("evaluation requires development data; test is locked")
    if len({(r["condition"], r["key"]) for r in rows}) != len(rows):
        raise ValueError("duplicate evaluation rows")
    if len({r["condition"] for r in rows}) != 1:
        raise ValueError("evaluate each condition separately")
    available, overlap, spatial = [], [], []
    for row in rows:
        selected = sensor or corrected_dms_winner(*(
            None if row[s]["candidate"] is None else row[s]["candidate"]["spatial_uncertainty_corrected"]
            for s in ("rgb", "thermal")))
        item = row[selected]["candidate"] if selected is not None else None
        box = None if item is None else Candidate(selected, tuple(item["box_xywh_native"]))
        frame = {"paired_presence": row["paired_presence"], **row["ground_truth"]}
        available.append(box is not None)
        overlap.append(frame_iou(frame, box))
        spatial.append(0. if item is None else item["spatial_uncertainty_corrected"])
    sequences, inverse = np.unique([r["sequence"] for r in rows], return_inverse=True)
    return {"label": np.array([r["paired_presence"] for r in rows], dtype=bool),
            "available": np.array(available), "iou": np.array(overlap), "spatial": np.array(spatial),
            "ids": np.array([r["key"] for r in rows]), "sequences": sequences,
            "inverse": inverse, "attributes": [r["attributes"] for r in rows]}


def sequence_scores(data, probability, threshold=0.5):
    p = np.asarray(probability)
    accepted = data["available"] & (p > 0.5) & (p >= threshold)
    y, group = data["label"], data["inverse"]
    iou = np.where(accepted, data["iou"], 0.)
    size = np.bincount(group)
    present = np.bincount(group, weights=y)
    failures = np.bincount(group, weights=y & (iou <= 0))
    fraction = np.divide(failures, present, out=np.zeros_like(present), where=present > 0)
    reward = np.bincount(group, weights=np.where(y, iou, ~accepted))
    return reward / size - 0.2 * fraction ** 0.3, accepted


def select_threshold(data, probability):
    scores = [(float(sequence_scores(data, probability, t)[0].mean()), t)
              for t in np.linspace(0.5, 0.95, 10)]
    return max(scores, key=lambda pair: (pair[0], -pair[1]))[1]


def reliability(probability, target, bins=10):
    p, y = np.asarray(probability), np.asarray(target)
    indices = np.minimum((p * bins).astype(int), bins - 1)
    result = []
    for index in range(bins):
        mask = indices == index
        result.append({"lower": index / bins, "upper": (index + 1) / bins,
                       "count": int(mask.sum()), "mean_probability": float(p[mask].mean()) if mask.any() else None,
                       "frequency": float(y[mask].mean()) if mask.any() else None})
    return result


def error_ranking(errors, score):
    from sklearn.metrics import average_precision_score, roc_auc_score
    errors, score = np.asarray(errors, dtype=bool), np.asarray(score)
    support = {"count": len(errors), "errors": int(errors.sum()),
               "prevalence": float(errors.mean()) if len(errors) else None}
    if len(np.unique(errors)) != 2:
        return {**support, "auroc": None, "auprc": None}
    return {**support, "auroc": float(roc_auc_score(errors, score)),
            "auprc": float(average_precision_score(errors, score))}


def average_precision(data, probability, iou_threshold):
    positives = int(data["label"].sum())
    if not positives:
        return None
    indices = np.flatnonzero(data["available"])
    if not len(indices):
        return 0.
    order = indices[np.lexsort((data["ids"][indices], -np.asarray(probability)[indices]))]
    tp = data["label"][order] & (data["iou"][order] >= iou_threshold)
    recall = np.cumsum(tp) / positives
    precision = np.cumsum(tp) / np.arange(1, len(order) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    slots = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.where(slots < len(order), envelope[np.minimum(slots, len(order) - 1)], 0).mean())


def evaluate(data, probability, uncertainty=None, *, threshold=0.5, box_scores=True):
    p, y = np.asarray(probability, dtype=float), data["label"]
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid model probabilities")
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty, dtype=float)
        if uncertainty.shape != y.shape or not np.isfinite(uncertainty).all() or np.any((uncertainty < 0) | (uncertainty > 1)):
            raise ValueError("invalid evidential uncertainty")
    prediction = p > 0.5
    correct = prediction == y
    confidence = np.maximum(p, 1 - p)
    bins = reliability(confidence, correct)
    positive_recall = float(prediction[y].mean()) if y.any() else None
    specificity = float((~prediction[~y]).mean()) if (~y).any() else None
    result = {"frames": len(y), "present": int(y.sum()), "absent": int((~y).sum()),
              "brier": float(np.mean((p - y) ** 2)), "presence_accuracy": float(correct.mean()),
              "sensitivity": positive_recall, "specificity": specificity,
              "balanced_accuracy": (positive_recall + specificity) / 2 if positive_recall is not None and specificity is not None else None,
              "ece": sum(b["count"] * abs(b["mean_probability"] - b["frequency"]) for b in bins if b["count"]) / len(y),
              "reliability": {"maximum_class": bins, "uav": reliability(p, y), "background": reliability(1 - p, ~y)},
              "presence_error_ranking": {"confidence": error_ranking(~correct, 1 - confidence)}}
    if uncertainty is not None:
        result["presence_error_ranking"]["evidential"] = error_ranking(~correct, uncertainty)
    result["brier_support"] = {}
    for name, mask in {"present": y, "absent": ~y, "available": data["available"], "missing": ~data["available"]}.items():
        result["brier_support"][name] = {"count": int(mask.sum()), "brier": float(np.mean((p[mask] - y[mask]) ** 2)) if mask.any() else None}
    if not box_scores:
        return result
    scores, accepted = sequence_scores(data, p, threshold)
    output_iou = np.where(accepted, data["iou"], 0.)
    result.update({"threshold": float(threshold), "edtc_sequence_mean": float(scores.mean()),
                   "edtc_frame_weighted": float(np.average(scores, weights=np.bincount(data["inverse"]))),
                   "per_sequence_edtc": dict(zip(data["sequences"].tolist(), scores.tolist())),
                   "pre_acceptance_miou": float(data["iou"][y].mean()) if y.any() else None,
                   "post_acceptance_miou": float(output_iou[y].mean()) if y.any() else None,
                   "accepted": int(accepted.sum()), "absent_outputs": int((accepted & ~y).sum()),
                   "present_no_output": int((~accepted & y).sum()),
                   "zero_overlap_failures": int((y & (output_iou <= 0)).sum())})
    ap = [average_precision(data, p, t) for t in np.linspace(0.5, 0.95, 10)]
    result.update({"ap50": ap[0], "map50_95": float(np.mean(ap)) if ap[0] is not None else None})
    result["localization_error_ranking"] = {}
    for name, mask, iou in (("all_present", y, output_iou), ("available_pre_acceptance", y & data["available"], data["iou"]),
                            ("accepted_present", y & accepted, output_iou)):
        rank = error_ranking(iou[mask] < 0.5, (1 - confidence)[mask])
        _, spatial_rank = np.unique(data["spatial"], return_inverse=True)
        spatial_rank[~accepted if name == "all_present" else ~data["available"]] = len(spatial_rank) + 1
        result["localization_error_ranking"][name] = {"confidence": rank,
            "spatial": error_ranking(iou[mask] < 0.5, spatial_rank[mask])}
        if uncertainty is not None:
            result["localization_error_ranking"][name]["evidential"] = error_ranking(iou[mask] < 0.5, uncertainty[mask])
    return result


def paired_bootstrap(first, second, *, replicates=2000, seed=11):
    if set(first) != set(second) or not first:
        raise ValueError("bootstrap requires identical nonempty sequence membership")
    difference = np.array([first[k] - second[k] for k in sorted(first)])
    indices = np.random.default_rng(seed).integers(0, len(difference), size=(replicates, len(difference)))
    interval = np.percentile(difference[indices].mean(axis=1), [2.5, 97.5])
    return {"difference": float(difference.mean()), "interval95": interval.tolist(),
            "sequences": len(difference), "replicates": replicates}


def oracle_acceptance(data):
    """Annotation-assisted detection bound for the fixed spatially selected boxes."""
    probability = (data["label"] & (data["iou"] > 0)).astype(float)
    scores, accepted = sequence_scores(data, probability)
    return {"edtc_sequence_mean": float(scores.mean()),
            "edtc_frame_weighted": float(np.average(scores, weights=np.bincount(data["inverse"]))),
            "present_frame_miou": float(data["iou"][data["label"]].mean()) if data["label"].any() else None,
            "accepted": int(accepted.sum()), "annotation_assisted": True,
            "scope": "Acceptance only; no sensor switching or box changes"}


def robustness_summary(conditions, corruption_ids):
    if len(corruption_ids) != 30 or len(set(corruption_ids)) != 30 or "clean" not in conditions:
        raise ValueError("require clean and exactly 30 distinct corruption conditions")
    if any(c not in conditions for c in corruption_ids):
        raise ValueError("incomplete robustness condition coverage")
    result = {"condition_count": 30, "clean": conditions["clean"], "metrics": {}}
    for name in ("edtc_sequence_mean", "brier", "ece", "ap50", "map50_95"):
        clean = conditions["clean"].get(name)
        supported = {c: conditions[c].get(name) for c in corruption_ids
                     if conditions[c].get(name) is not None}
        result["metrics"][name] = {
            "supported_conditions": len(supported),
            "mean": float(np.mean(list(supported.values()))) if supported else None,
            "delta_by_condition": {c: float(v - clean) if clean is not None else None for c, v in supported.items()}}
    result["dropouts"] = {c: v for c, v in conditions.items() if "dropout" in c}
    return result


def bootstrap_edtc(first, second, conditions, *, replicates=2000, seed=11):
    """Paired sequence resampling, retaining all seeds and conditions together.

    Inputs: seed -> condition -> sequence -> EDTC. Seeds are fixed fitted models;
    they are averaged, not resampled. This helper is not valid for nonlinear
    pooled metrics such as AP, Brier ranking or AUROC.
    """
    if not first or set(first) != set(second) or not conditions or replicates < 1:
        raise ValueError("identical seeds, conditions and positive replicate count required")
    seeds = sorted(first)
    sequences = None
    differences = []
    for s in seeds:
        by_condition = []
        for c in conditions:
            if c not in first[s] or c not in second[s]:
                raise ValueError("missing paired condition")
            a, b = first[s][c], second[s][c]
            if not a or set(a) != set(b) or (sequences is not None and set(a) != set(sequences)):
                raise ValueError("identical sequence coverage required")
            sequences = sorted(a)
            delta = np.array([a[k] - b[k] for k in sequences], dtype=float)
            if not np.isfinite(delta).all():
                raise ValueError("undefined EDTC input; report unsupported comparisons separately")
            by_condition.append(delta)
        differences.append(by_condition)
    delta = np.mean(np.array(differences), axis=(0, 1))
    indices = np.random.default_rng(seed).integers(0, len(delta), size=(replicates, len(delta)))
    interval = np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])
    return {"difference": float(delta.mean()), "interval95": interval.tolist(), "replicates": replicates,
            "sequences": len(delta), "seeds": seeds, "conditions": list(conditions), "undefined_replicates": 0}
