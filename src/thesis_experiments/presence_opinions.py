"""Study-specific frame opinions; independent of inherited candidate filtering.

Torch is imported only by the feature adapter, model and training loss.
Class order is always [UAV, background]; annotation presence is a boolean.
"""

import hashlib

import numpy as np


def sample_by_class(frames, sensor, per_class, *, split, seed=11):
    """Predeclare a bounded, reproducible sample without inspecting predictions."""
    if sensor not in ("rgb", "thermal") or split not in ("train", "val") or per_class < 1:
        raise ValueError("invalid development sampling request")
    if any(row["split"] != split for row in frames):
        raise ValueError("mixed/test splits cannot enter opinion fitting or validation")
    if len({row["key"] for row in frames}) != len(frames):
        raise ValueError("duplicate frame key")
    result = []
    for present in (True, False):
        rows = [row for row in frames if row[sensor]["present"] is present]
        if not rows:
            raise ValueError(f"no {'UAV' if present else 'background'} support for {sensor}")
        rows.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{sensor}:{row['key']}".encode()).digest())
        result.extend(rows[:per_class])
    return sorted(result, key=lambda row: row["key"])


def fit_standardization(features, *, split):
    if split != "train":
        raise ValueError("opinion feature statistics must be fitted on training only")
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("invalid feature matrix")
    return values.mean(0), np.maximum(values.std(0), 1e-8)


def opinion(evidence):
    evidence = np.asarray(evidence, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] != 2 or not np.isfinite(evidence).all() or (evidence < 0).any():
        raise ValueError("expected finite nonnegative [N,2] evidence")
    strength = (evidence + 1).sum(1, keepdims=True)
    return {"beliefs": evidence / strength, "uncertainty": 2 / strength[:, 0],
            "probabilities": (evidence + 1) / strength}


def classification_report(evidence, presence, *, population_prevalence):
    """Natural-prevalence-weighted metrics for a class-stratified validation sample."""
    values = opinion(evidence)
    labels = np.asarray(presence, dtype=bool)
    if labels.shape != (len(evidence),) or len(np.unique(labels)) != 2:
        raise ValueError("report requires both validation classes")
    if not 0 < population_prevalence < 1:
        raise ValueError("invalid population prevalence")
    probability = values["probabilities"][:, 0]
    prediction = probability > 0.5
    weights = np.where(labels, population_prevalence / labels.sum(),
                       (1 - population_prevalence) / (~labels).sum())
    positive_recall = float(prediction[labels].mean())
    negative_recall = float((~prediction[~labels]).mean())
    brier = float((weights * (probability - labels) ** 2).sum())
    return {"sample_uav": int(labels.sum()), "sample_background": int((~labels).sum()),
            "uav_recall": positive_recall, "background_recall": negative_recall,
            "balanced_accuracy": (positive_recall + negative_recall) / 2,
            "population_weighted_brier": brier,
            "constant_prevalence_brier": population_prevalence * (1 - population_prevalence),
            "u_mean_sample": float(values["uncertainty"].mean()),
            "both_class_learning_gate": bool(positive_recall > 0.5 and negative_recall > 0.5
                and brier < population_prevalence * (1 - population_prevalence))}


def paired_report(rgb_evidence, thermal_evidence):
    rgb, thermal = opinion(rgb_evidence), opinion(thermal_evidence)
    if rgb["beliefs"].shape != thermal["beliefs"].shape:
        raise ValueError("opinions must be paired by identical frame keys")
    br, bt = rgb["beliefs"], thermal["beliefs"]
    conflict = br[:, 0] * bt[:, 1] + br[:, 1] * bt[:, 0]
    degree = (0.5 * np.abs(rgb["probabilities"] - thermal["probabilities"]).sum(1)
              * (1 - rgb["uncertainty"]) * (1 - thermal["uncertainty"]))
    pr, pt = rgb["probabilities"][:, 0], thermal["probabilities"][:, 0]
    opposite = ((pr > 0.5) & (pt < 0.5)) | ((pt > 0.5) & (pr < 0.5))
    # Fixed diagnostic threshold, not tuned until conflict appears.
    confident_opposite = ((pr >= 0.7) & (pt <= 0.3)) | ((pt >= 0.7) & (pr <= 0.3))
    return {"pairs": len(pr), "opposite_decisions": int(opposite.sum()),
            "confident_opposite_decisions": int(confident_opposite.sum()),
            "inherited_conflict_mean": float(conflict.mean()),
            "inherited_conflict_max": float(conflict.max()),
            "degree_of_conflict_mean": float(degree.mean()),
            "degree_of_conflict_max": float(degree.max())}


def build_head(input_size, hidden_size=16):
    import torch
    from torch import nn
    head = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                         nn.Linear(hidden_size, 2), nn.Softplus())
    for layer in head:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return head


def evidential_loss(evidence, presence, *, epoch, population_prevalence):
    """Expected CE + annealed KL; importance weighting restores the training prior."""
    import torch
    if evidence.ndim != 2 or evidence.shape[1] != 2 or epoch < 1:
        raise ValueError("invalid evidential loss inputs")
    labels = presence.to(dtype=torch.bool)
    if labels.shape != evidence.shape[:1] or not labels.any() or labels.all():
        raise ValueError("pilot training batch must contain both classes")
    if not 0 < population_prevalence < 1:
        raise ValueError("invalid training prevalence")
    target = torch.stack((labels, ~labels), dim=1).to(evidence.dtype)
    alpha = evidence + 1
    ce = (target * (torch.digamma(alpha.sum(1, keepdim=True)) - torch.digamma(alpha))).sum(1)
    adjusted = target + (1 - target) * alpha
    kl = torch.distributions.kl_divergence(torch.distributions.Dirichlet(adjusted),
                                         torch.distributions.Dirichlet(torch.ones_like(adjusted)))
    weights = torch.where(labels, population_prevalence / labels.sum(),
                          (1 - population_prevalence) / (~labels).sum())
    return ((ce + min(1.0, epoch / 10) * kl) * weights).sum()


class FrozenFrameFeatures:
    """Pool the frozen neck maps entering Detect; no box or evidence filtering.

    Detector remains in eval mode, including batch normalization. The hook is
    removed on close and never modifies the inherited source or output.
    """

    def __init__(self, detector):
        import torch
        self.torch = torch
        self.detector = detector.eval()
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        self.value = None
        self.handle = detector.model[-1].register_forward_pre_hook(self._capture)

    def _capture(self, _module, inputs):
        self.value = self.torch.cat([
            pooled for feature in inputs[0]
            for pooled in (feature.mean(dim=(-2, -1)), feature.amax(dim=(-2, -1)))
        ], dim=1)

    def __call__(self, images):
        if self.detector.training:
            raise RuntimeError("frozen feature extractor must remain in eval mode")
        self.value = None
        with self.torch.no_grad():
            self.detector(images)
        if self.value is None or not self.torch.isfinite(self.value).all():
            raise RuntimeError("invalid or missing detector features")
        return self.value.detach()

    def close(self):
        self.handle.remove()
