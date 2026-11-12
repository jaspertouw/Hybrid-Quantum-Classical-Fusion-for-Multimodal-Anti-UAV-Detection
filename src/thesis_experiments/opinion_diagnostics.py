"""Clean-only support diagnostics and reusable frozen-neck feature artifacts."""

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from . import presence_opinions as opinions
from .detector_cache import digest, immutable_json
from .provenance import sha256_file


def support(rows, sensor):
    result = {}
    for present, label in ((True, "uav"), (False, "background")):
        counts = Counter(row["sequence"] for row in rows if row[sensor]["present"] is present)
        total = sum(counts.values())
        result[label] = {"frames": total, "sequences": len(counts),
                         "largest_sequence_fraction": max(counts.values(), default=0) / max(total, 1),
                         "per_sequence": dict(sorted(counts.items()))}
    return result


def feature_context(rows, sensor, identities):
    splits = {row["split"] for row in rows}
    if len(splits) != 1 or not splits <= {"train", "val"}:
        raise ValueError("features require a single development split")
    if sensor not in ("rgb", "thermal") or len({row["key"] for row in rows}) != len(rows):
        raise ValueError("invalid feature membership")
    return {"schema_version": 1, "sensor": sensor, "split": next(iter(splits)),
            "condition": "clean", "identities": identities,
            "frames": [{"key": row["key"], "sequence": row["sequence"],
                        "present": row[sensor]["present"], "image": row[sensor]["image"]} for row in rows]}


def cached_features(root, context, extract):
    """Commit one complete NPZ/metadata directory atomically; verify before reuse."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / digest(context)
    if not path.exists():
        values = np.asarray(extract(), dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(context["frames"]) or not np.isfinite(values).all():
            raise ValueError("invalid extracted feature matrix")
        staging = Path(tempfile.mkdtemp(prefix=".partial-", dir=root))
        try:
            np.savez(staging / "features.npz", values=values)
            immutable_json(staging / "COMPLETED.json", {"context": context,
                "features_sha256": sha256_file(staging / "features.npz"), "shape": list(values.shape)})
            try:
                os.rename(staging, path)
            except OSError:
                if not path.exists():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    record = json.loads((path / "COMPLETED.json").read_text())
    if record["context"] != context or sha256_file(path / "features.npz") != record["features_sha256"]:
        raise ValueError("feature cache provenance/hash mismatch")
    with np.load(path / "features.npz", allow_pickle=False) as artifact:
        values = artifact["values"]
    if values.ndim != 2 or list(values.shape) != record["shape"] or len(values) != len(context["frames"]) or not np.isfinite(values).all():
        raise ValueError("feature cache shape/value mismatch")
    return values, {"path": str(path.resolve()), "features_sha256": record["features_sha256"],
                    "context_sha256": digest(context)}


def subset_features(values, rows, subset):
    indices = {row["key"]: index for index, row in enumerate(rows)}
    if len(indices) != len(rows) or len(values) != len(rows):
        raise ValueError("feature membership mismatch")
    return values[[indices[row["key"]] for row in subset]]


def class_losses(evidence, labels, epoch):
    """Unweighted per-class expected CE and KL, without changing training loss."""
    import torch
    alpha = evidence + 1
    target = torch.stack((labels, ~labels), 1).to(evidence.dtype)
    ce = (target * (torch.digamma(alpha.sum(1, keepdim=True)) - torch.digamma(alpha))).sum(1)
    adjusted = target + (1 - target) * alpha
    kl = torch.distributions.kl_divergence(torch.distributions.Dirichlet(adjusted),
                                         torch.distributions.Dirichlet(torch.ones_like(adjusted)))
    return {name: {"expected_ce": float(ce[mask].mean()), "kl": float(kl[mask].mean()),
                   "total": float((ce + min(1, epoch / 10) * kl)[mask].mean())}
            for name, mask in (("uav", labels), ("background", ~labels))}


def per_sequence_report(evidence, rows, sensor):
    prediction = opinions.opinion(evidence)["probabilities"][:, 0] > 0.5
    counts = {}
    for predicted, row in zip(prediction, rows):
        report = counts.setdefault(row["sequence"], {"uav": 0, "background": 0,
                                                    "uav_correct": 0, "background_correct": 0})
        present = row[sensor]["present"]
        label = "uav" if present else "background"
        report[label] += 1
        report[f"{label}_correct"] += int(predicted == present)
    return counts


def train_arm(train_values, train_rows, val_values, val_rows, sensor, spec, prevalence, val_prevalence):
    """Fixed 50-epoch, full-batch objective, same initialization for both arms."""
    import torch
    torch.manual_seed(spec["seed"])
    mean, scale = opinions.fit_standardization(train_values, split="train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = lambda values: torch.from_numpy(((values - mean) / scale).astype(np.float32)).to(device)
    x, vx = transform(train_values), transform(val_values)
    labels = torch.tensor([row[sensor]["present"] for row in train_rows], device=device)
    head = opinions.build_head(x.shape[1], spec["hidden_size"]).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=spec["learning_rate"])
    history = []
    for epoch in range(1, spec["epochs"] + 1):
        head.train()
        optimizer.zero_grad()
        loss = opinions.evidential_loss(head(x), labels, epoch=epoch, population_prevalence=prevalence)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite diagnostic loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 10)
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                history.append({"epoch": epoch, "pre_update_loss": float(loss.detach()),
                                "post_update_train_class_losses": class_losses(head(x), labels, epoch)})
    head.eval()
    with torch.no_grad():
        training_evidence, validation_evidence = head(x).cpu().numpy(), head(vx).cpu().numpy()
    report = {"train": opinions.classification_report(training_evidence, labels.cpu().numpy(),
                                                        population_prevalence=prevalence),
              "validation": opinions.classification_report(validation_evidence,
                  [row[sensor]["present"] for row in val_rows], population_prevalence=val_prevalence),
              "training_class_history": history,
              "validation_per_sequence": per_sequence_report(validation_evidence, val_rows, sensor),
              "training_per_sequence": per_sequence_report(training_evidence, train_rows, sensor),
              "train_keys": [row["key"] for row in train_rows],
              "validation_keys": [row["key"] for row in val_rows]}
    checkpoint = {"state_dict": head.cpu().state_dict(), "mean": mean.tolist(), "scale": scale.tolist(),
                  "input_size": x.shape[1], "status": "diagnostic_only", "sensor": sensor,
                  "specification": spec, "train_keys": report["train_keys"]}
    return report, checkpoint, validation_evidence
