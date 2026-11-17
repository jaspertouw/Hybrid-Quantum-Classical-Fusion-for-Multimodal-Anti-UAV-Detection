"""Independent numerical and cache-to-result checks, never a research model."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import numpy as np
import torch

from . import detector_cache as cache
from .features import fit_features, transform_features
from .fusion import predict_reference
from .metrics import evaluate, evaluation_data
from .models import ClassicalModel
from .neural_fusion import NeuralFusion, train_neural


CLEAN = {"id": "clean", "sensor": None, "corruption": "clean", "severity": None}


def synthetic_cache(root, split, sequences):
    frames = defaultdict(list)
    for sequence in sequences:
        for number in range(1, 31):
            present = number % 5 not in (0, 3)
            annotation = {"present": present, "box_valid": present,
                          "box_xywh": [0, 0, 10, 10] if present else None}
            frames[sequence].append({"key": f"{split}/{sequence}/{number:06d}", "split": split,
                "sequence": sequence, "frame": number, "paired_presence": present, "attributes": ["LI", "TC"],
                "rgb": dict(annotation), "thermal": dict(annotation)})
    context = {"purpose": "synthetic-verification-only", "conditions": [CLEAN],
               "selected_keys_sha256": cache.digest(sorted(f["key"] for group in frames.values() for f in group))}
    identity = cache.initialize(root, context)
    for sequence, group in frames.items():
        rows = []
        for frame in group:
            sensors = {}
            for sensor in ("rgb", "thermal"):
                missing = frame["frame"] % 5 == 0 or (sensor == "rgb" and frame["frame"] % 5 == 2)
                belief = [0.7, 0.1] if frame["paired_presence"] else [0.3, 0.4]
                spatial = 0.1 if sensor == "rgb" else 0.2
                candidate = None if missing else {
                    "beliefs": belief, "semantic_uncertainty": 1 - sum(belief), "objectness": 0.9,
                    "candidate_score": 0.9 * belief[0], "anchor_index": 1,
                    "box_xywh_input": [5, 5, 10, 10], "box_xywh_native": [0, 0, 10, 10],
                    "sigma2_grid": [100 * spatial / 64] * 4, "sigma2_pixel": [100 * spatial] * 4,
                    "sigma2_norm_corrected_components": [spatial] * 4,
                    "sigma2_norm_inherited_head_components": [spatial] * 4,
                    "spatial_uncertainty_corrected": spatial}
                sensors[sensor] = cache.sensor_record(candidate)
            rows.append(cache.make_row(frame, CLEAN, sensors, identity))
        cache.write_shard(cache.shard_path(root, CLEAN, sequence), rows, CLEAN, identity, group)
    cache.finalize(root, identity, frames, [CLEAN])


def synthetic_run(root, seed=11):
    root = Path(root)
    synthetic_cache(root / "train", "train", [f"fit{i}" for i in range(6)])
    synthetic_cache(root / "val", "val", [f"heldout{i}" for i in range(3)])
    training, validation = list(cache.iter_cache(root / "train")), list(cache.iter_cache(root / "val"))
    fitted = fit_features(training)
    raw, normalized, _ = transform_features(training, fitted)
    vr, vn, _ = transform_features(validation, fitted)
    y = np.array([r["paired_presence"] for r in training])
    groups = np.array([r["sequence"] for r in training])
    data = evaluation_data(validation)
    predictions, results, histories = {}, {}, {}
    for method in ("always_present", "prevalence", "rgb", "thermal", "naive", "abf", "dbf"):
        p, u = predict_reference(validation, method, prevalence=float(y.mean()))
        own_data = evaluation_data(validation, method) if method in ("rgb", "thermal") else data
        results[method] = evaluate(own_data, p, u, box_scores=method not in ("always_present", "prevalence"))
        predictions[method] = p
    for kind in ("svm", "forest"):
        settings = {"C": 1., "gamma": 0.1} if kind == "svm" else {"max_depth": 4, "min_samples_leaf": 1}
        model = ClassicalModel(kind, **settings).fit(raw, y, groups)
        p, u = model.predict(vr)
        model.save(root / f"{kind}.joblib")
        restored = model.load(root / f"{kind}.joblib").predict(vr)[0]
        np.testing.assert_allclose(p, restored, rtol=0, atol=1e-12)
        predictions[kind], results[kind] = p, evaluate(data, p, u)
    for kind in ("gate", "mlp", "vqc"):
        torch.manual_seed(seed)
        model = NeuralFusion(kind)
        tx, vx = (raw, vr) if kind == "gate" else (normalized, vn)
        history = train_neural(model, tx, y, vx, data, epochs=3, seed=seed)
        histories[kind] = {k: v for k, v in history.items() if k != "seconds"}
        p, u = model.predict(vx)
        model.save(root / f"{kind}.pt")
        np.testing.assert_allclose(p, model.load(root / f"{kind}.pt").predict(vx)[0], rtol=0, atol=1e-12)
        predictions[kind], results[kind] = p, evaluate(data, p, u)
    return {"features": fitted, "metrics": results, "training": histories}, predictions


def dense_quantum_reference(x, angles, layout="ring", reupload=True):
    """Independent NumPy Kronecker matrices, not Torch gate/index helpers."""
    state = np.zeros(16, dtype=complex)
    state[0] = 1

    def apply(axis, theta, wire):
        nonlocal state
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        if axis == "x":
            local = np.array([[c, -1j * s], [-1j * s, c]])
        elif axis == "y":
            local = np.array([[c, -s], [s, c]])
        else:
            local = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
        full = np.kron(np.eye(2 ** wire), np.kron(local, np.eye(2 ** (3 - wire))))
        state = full @ state

    for block in range(len(angles)):
        if block == 0 or reupload:
            for wire in range(4):
                apply("y", math.pi / 4 + math.pi / 2 * x[wire], wire)
                apply("z", math.pi * x[wire + 4], wire)
        for wire in range(4):
            for axis, theta in zip(("x", "z", "y"), angles[block, wire]):
                apply(axis, theta, wire)
        edges = [] if layout == "none" else [(0, 1), (1, 2), (2, 3)]
        if layout == "ring":
            edges.append((3, 0))
        for control, target in edges:
            matrix = np.zeros((16, 16))
            for index in range(16):
                bits = list(f"{index:04b}")
                if bits[control] == "1":
                    bits[target] = "0" if bits[target] == "1" else "1"
                matrix[int("".join(bits), 2), index] = 1
            state = matrix @ state
    probability = np.abs(state) ** 2
    return np.array([sum(probability[i] * (1 if f"{i:04b}"[q] == "0" else -1)
                         for i in range(16)) for q in range(4)])


def verify_quantum_reference():
    rng = np.random.default_rng(11)
    errors = []
    for depth in (1, 2, 4):
        for layout in ("ring", "chain", "none"):
            for reupload in (True, False):
                torch.manual_seed(11)
                model = NeuralFusion("vqc", depth=depth, layout=layout, reupload=reupload)
                x = rng.uniform(size=(2, 8))
                with torch.no_grad():
                    actual = model.quantum_features(torch.tensor(x, dtype=torch.float64)).numpy()
                expected = np.array([dense_quantum_reference(row, model.angles.detach().numpy(), layout, reupload) for row in x])
                errors.append(float(np.max(np.abs(actual - expected))))
    if max(errors) > 1e-12:
        raise AssertionError("Torch statevector and independent dense reference disagree")
    return {"cases": 36, "maximum_absolute_error": max(errors), "tolerance": 1e-12,
            "reference": "independent NumPy full 16x16 Kronecker unitaries"}
