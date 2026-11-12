"""Training-sequence holdout study of frame-head class weighting only."""

import hashlib
import json
from pathlib import Path

import numpy as np

from . import opinion_diagnostics as diagnostic
from . import presence_opinions as opinions
from .detector_cache import digest
from .provenance import sha256_file


def training_sequence_split(population, rows, sensor, *, seed=11):
    """One third held out, stratifying sequences by any background support.

    Uses training annotations only. No retrying splits based on model outcomes.
    """
    if any(row["split"] != "train" for row in rows):
        raise ValueError("objective study accepts training frames only")
    positives = population["uav"]["per_sequence"]
    negatives = population["background"]["per_sequence"]
    sequences = set(positives) | set(negatives)
    if not {row["sequence"] for row in rows} <= sequences:
        raise ValueError("unknown training sequence")
    heldout = set()
    for has_background in (True, False):
        group = [seq for seq in sequences if bool(negatives.get(seq, 0)) == has_background]
        group.sort(key=lambda seq: hashlib.sha256(f"{seed}:objective-split:{seq}".encode()).digest())
        heldout.update(group[::3])
    fitting = sequences - heldout
    selected = {"fit": [row for row in rows if row["sequence"] in fitting],
                "heldout_train": [row for row in rows if row["sequence"] in heldout]}
    prevalences = {}
    for name, seqs in (("fit", fitting), ("heldout_train", heldout)):
        counts = {"uav": sum(positives.get(seq, 0) for seq in seqs),
                  "background": sum(negatives.get(seq, 0) for seq in seqs)}
        if not all(counts.values()) or len({row[sensor]["present"] for row in selected[name]}) != 2:
            raise ValueError("fixed partition lacks both-class support; no automatic split search")
        prevalences[name] = counts["uav"] / sum(counts.values())
    return selected, {"fit_sequences": sorted(fitting), "heldout_train_sequences": sorted(heldout),
                      "population_prevalence": prevalences}


def read_training_features(record, sensor, phase2_sha256, repository):
    path = Path(record["path"])
    metadata = json.loads((path / "COMPLETED.json").read_text())
    context = metadata["context"]
    if context["split"] != "train" or context["sensor"] != sensor or context["condition"] != "clean":
        raise ValueError("objective study cannot read validation/test/corrupted features")
    if context["identities"]["phase2_sha256"] != phase2_sha256 or digest(context) != record["context_sha256"]:
        raise ValueError("training feature context mismatch")
    for relative, expected in context["identities"]["source_sha256"].items():
        if sha256_file(repository / relative) != expected:
            raise ValueError(f"cached feature source changed: {relative}")
    expected = record["features_sha256"]
    if metadata["features_sha256"] != expected or sha256_file(path / "features.npz") != expected:
        raise ValueError("training feature hash mismatch")
    with np.load(path / "features.npz", allow_pickle=False) as archive:
        values = archive["values"]
    if values.ndim != 2 or list(values.shape) != metadata["shape"] or len(values) != len(context["frames"]) or not np.isfinite(values).all():
        raise ValueError("invalid training feature matrix")
    rows = [{"key": row["key"], "sequence": row["sequence"], "split": "train",
             sensor: {"present": row["present"], "image": row["image"]}} for row in context["frames"]]
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError("duplicate training keys")
    return values, rows


def probability_report(evidence, rows, sensor, population_prevalence, fitting_prevalence):
    labels = np.array([row[sensor]["present"] for row in rows], dtype=bool)
    report = opinions.classification_report(evidence, labels, population_prevalence=population_prevalence)
    probability = opinions.opinion(evidence)["probabilities"][:, 0]
    weights = np.where(labels, population_prevalence / labels.sum(),
                       (1 - population_prevalence) / (~labels).sum())
    probability = np.clip(probability, 1e-8, 1 - 1e-8)
    report["population_weighted_log_loss"] = float(-(weights * np.where(labels,
        np.log(probability), np.log1p(-probability))).sum())
    report["constant_fitting_prevalence_brier"] = (population_prevalence * (1 - fitting_prevalence) ** 2
                                               + (1 - population_prevalence) * fitting_prevalence ** 2)
    report["constant_fitting_prevalence_log_loss"] = float(-population_prevalence * np.log(fitting_prevalence)
                                            - (1 - population_prevalence) * np.log1p(-fitting_prevalence))
    report["uav_mean_p_uav"] = float(probability[labels].mean())
    report["background_mean_p_uav"] = float(probability[~labels].mean())
    return report


def fit_objective(values, rows, selected, split, sensor, head_spec, objective):
    if objective not in ("natural_prevalence", "class_balanced"):
        raise ValueError("undeclared objective")
    fitting, heldout = selected["fit"], selected["heldout_train"]
    if any(row["split"] != "train" for row in fitting + heldout):
        raise ValueError("training-only comparison")
    if {row["sequence"] for row in fitting} & {row["sequence"] for row in heldout}:
        raise ValueError("sequence leakage")
    train_values = diagnostic.subset_features(values, rows, fitting)
    heldout_values = diagnostic.subset_features(values, rows, heldout)
    pi_fit, pi_hold = (split["population_prevalence"][name] for name in ("fit", "heldout_train"))
    target_prevalence = pi_fit if objective == "natural_prevalence" else 0.5
    report, checkpoint, evidence = diagnostic.train_arm(train_values, fitting,
        heldout_values, heldout, sensor, head_spec, target_prevalence, pi_hold)
    # Training objective's artificial class prior must not leak into reported population metrics.
    import torch
    head = opinions.build_head(checkpoint["input_size"], head_spec["hidden_size"])
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()
    normalized = ((train_values - np.array(checkpoint["mean"])) / np.array(checkpoint["scale"])).astype(np.float32)
    with torch.no_grad():
        train_evidence = head(torch.from_numpy(normalized)).numpy()
    result = {"objective": objective, "objective_uav_weight": target_prevalence,
              "train": probability_report(train_evidence, fitting, sensor, pi_fit, pi_fit),
              "heldout_train": probability_report(evidence, heldout, sensor, pi_hold, pi_fit),
              "training_class_history": report["training_class_history"],
              "heldout_train_per_sequence": report["validation_per_sequence"],
              "fit_keys": report["train_keys"], "heldout_train_keys": report["validation_keys"]}
    checkpoint["objective"] = objective
    checkpoint["objective_uav_weight"] = target_prevalence
    return result, checkpoint
