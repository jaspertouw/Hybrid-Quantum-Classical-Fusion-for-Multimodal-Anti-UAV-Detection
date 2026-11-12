"""Positive classwise evidence scaling fitted on separate training sequences.

This preserves nonnegative evidence and the original Dirichlet identities.
Probability calibration does not establish calibrated evidential uncertainty.
"""

import hashlib

import numpy as np

from .presence_opinions import opinion


def calibration_partition(population, rows, original_partition, sensor, *, seed=11):
    if any(row["split"] != "train" for row in rows):
        raise ValueError("calibration study uses original training frames only")
    fitting = set(original_partition["fit_sequences"])
    heldout = set(original_partition["heldout_train_sequences"])
    if fitting & heldout or fitting | heldout != set(population["uav"]["per_sequence"]) | set(population["background"]["per_sequence"]):
        raise ValueError("invalid original training partition")
    calibration = set()
    background = population["background"]["per_sequence"]
    for has_background in (True, False):
        group = [seq for seq in heldout if bool(background.get(seq, 0)) == has_background]
        group.sort(key=lambda seq: hashlib.sha256(f"{seed}:calibration-split:{seq}".encode()).digest())
        calibration.update(group[::2])
    evaluation = heldout - calibration
    groups = {"head_fit": fitting, "calibration": calibration, "evaluation": evaluation}
    selected, priors = {}, {}
    for name, sequences in groups.items():
        selected[name] = [row for row in rows if row["sequence"] in sequences]
        positives = sum(population["uav"]["per_sequence"].get(seq, 0) for seq in sequences)
        negatives = sum(background.get(seq, 0) for seq in sequences)
        if not positives or not negatives or len({row[sensor]["present"] for row in selected[name]}) != 2:
            raise ValueError("fixed three-way split lacks class support; no split retries")
        priors[name] = positives / (positives + negatives)
    return selected, {"sequences": {name: sorted(sequences) for name, sequences in groups.items()},
                      "population_prevalence": priors}


def apply_scaling(evidence, log_scales, *, bound=6):
    evidence = np.asarray(evidence, dtype=np.float64)
    opinion(evidence)  # shared finite/range/shape validation
    scales = np.asarray(log_scales, dtype=np.float64)
    if scales.shape != (2,) or not np.isfinite(scales).all() or (np.abs(scales) > bound + 1e-8).any():
        raise ValueError("invalid calibration scales")
    result = evidence * np.exp(scales)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite calibrated evidence")
    return result


def fit_scaling(evidence, presence, population_prevalence, *, regularization=0.001, bound=6, max_iter=100):
    """Two-parameter weighted NLL + identity penalty; no evaluation input."""
    import torch
    evidence = np.asarray(evidence, dtype=np.float64)
    opinion(evidence)
    labels = np.asarray(presence, dtype=bool)
    if labels.shape != (len(evidence),) or len(np.unique(labels)) != 2 or not 0 < population_prevalence < 1:
        raise ValueError("calibration needs both classes and a natural prior")
    if regularization <= 0 or bound <= 0 or max_iter < 1:
        raise ValueError("invalid predeclared calibration settings")
    values = torch.tensor(evidence, dtype=torch.float64)
    target = torch.tensor(np.where(labels, 0, 1), dtype=torch.long)
    weights = torch.tensor(np.where(labels, population_prevalence / labels.sum(),
                                    (1 - population_prevalence) / (~labels).sum()), dtype=torch.float64)
    raw = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def loss():
        theta = bound * torch.tanh(raw / bound)
        alpha = values * torch.exp(theta) + 1
        log_probability = torch.log(alpha) - torch.log(alpha.sum(1, keepdim=True))
        nll = -(weights * log_probability.gather(1, target[:, None]).squeeze(1)).sum()
        return nll + regularization * (theta ** 2).mean()

    def closure():
        optimizer.zero_grad()
        value = loss()
        if not torch.isfinite(value):
            raise RuntimeError("nonfinite calibration objective")
        value.backward()
        return value

    before = float(loss().detach())
    optimizer.step(closure)
    after = float(loss().detach())
    if after > before + 1e-8:
        raise RuntimeError("calibration optimization did not improve its fitting objective")
    theta = (bound * torch.tanh(raw / bound)).detach().numpy()
    return {"log_scales": theta.tolist(), "scales": np.exp(theta).tolist(),
            "regularization": regularization, "bound": bound, "max_iter": max_iter,
            "objective_before": before, "objective_after": after,
            "near_bound": bool((np.abs(theta) >= 0.95 * bound).any()),
            "uncertainty_calibration_claimed": False}


def joint_gate(raw, calibrated):
    return bool(calibrated["uav_recall"] > 0.5 and calibrated["background_recall"] > 0.5
                and calibrated["population_weighted_brier"] < raw["population_weighted_brier"]
                and calibrated["population_weighted_log_loss"] < raw["population_weighted_log_loss"]
                and calibrated["population_weighted_brier"] < calibrated["constant_fitting_prevalence_brier"])
