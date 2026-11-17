#!/usr/bin/env python3
"""Clean-development classical grids and the prescribed shared evidential-loss pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

import numpy as np
import torch

from scripts.successor.run_fusion_pilot import read_cache
from src.thesis_experiments.config import load_experiment_config
from src.thesis_experiments.features import fit_features, transform_features, training_sample
from src.thesis_experiments.fusion import predict_reference
from src.thesis_experiments.io import atomic_write_json
from src.thesis_experiments.metrics import evaluate, evaluation_data, select_threshold, sequence_scores
from src.thesis_experiments.models import ClassicalModel
from src.thesis_experiments.neural_fusion import NeuralFusion, train_neural
from src.thesis_experiments.run import RunLayout


def criterion(data, probability):
    return (float(sequence_scores(data, probability)[0].mean()),
            -float(np.mean((probability - data["label"]) ** 2)))


def choose_loss(results):
    """Methodology's predeclared rule, without using any corruption outcomes."""
    if any(results[k]["cross_entropy"].get("nonfinite", False) for k in ("mlp", "vqc")):
        if any(results[k]["squared_error"].get("nonfinite", False) for k in ("mlp", "vqc")):
            raise FloatingPointError("no common finite evidential loss")
        return "squared_error"
    if any(results[k]["squared_error"].get("nonfinite", False) for k in ("mlp", "vqc")):
        return "cross_entropy"
    improve = all(results[k]["squared_error"]["brier"] < results[k]["cross_entropy"]["brier"]
                  and results[k]["squared_error"]["reference_edtc"] >= results[k]["cross_entropy"]["reference_edtc"] - 0.01
                  for k in ("mlp", "vqc"))
    return "squared_error" if improve else "cross_entropy"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/caches/v1"))
    parser.add_argument("--run-id", default="clean-fusion-selection-v1")
    args = parser.parse_args()
    config = load_experiment_config()
    protocol = json.loads(config.protocol_path.read_text())
    run = RunLayout(config, args.run_id)
    if run.root.exists():
        raise FileExistsError("choose a fresh run ID")
    sources = [Path(__file__), config.protocol_path, *(REPOSITORY / "src/thesis_experiments" / file
        for file in ("features.py", "metrics.py", "fusion.py", "models.py", "neural_fusion.py"))]
    run.initialize(sources)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    train, training_record = read_cache(args.cache_root / "clean/train", "train", protocol)
    validation, validation_record = read_cache(args.cache_root / "clean/val", "val", protocol)
    for field in ("phase2_manifest_sha256", "development_manifest_sha256", "corruption_specification_sha256"):
        if training_record["context"][field] != validation_record["context"][field]:
            raise ValueError("training/validation contracts differ")
    fitted = fit_features(train)
    raw, normalized, _ = transform_features(train, fitted)
    val_raw, val_normalized, clipping = transform_features(validation, fitted)
    settings = protocol["training"]
    indices = training_sample(train, settings["optional_training_pair_cap"], settings["selection_seed"])
    labels = np.array([r["paired_presence"] for r in train], dtype=bool)[indices]
    groups = np.array([r["sequence"] for r in train])[indices]
    data = evaluation_data(validation)
    atomic_write_json(run.artifact_dir("models") / "features.json", fitted)
    atomic_write_json(run.artifact_dir("manifests") / "training_sample.json", {"keys": [train[i]["key"] for i in indices]})
    report = {"scope": "Clean development selection only; VQC/MLP architecture grids and final seeds remain pending",
              "test_accessed": False, "inputs": {"train": training_record, "validation": validation_record},
              "training_pairs": len(indices), "validation_clipping": clipping, "classical": {}, "loss_pilot": {}}
    grids = {"svm": [{"C": c, "gamma": g} for c in protocol["models"]["rbf_svm"]["C"] for g in protocol["models"]["rbf_svm"]["gamma"]],
             "forest": [{"max_depth": d, "min_samples_leaf": leaf} for d in protocol["models"]["random_forest"]["max_depth"]
                        for leaf in protocol["models"]["random_forest"]["min_samples_leaf"]]}
    for kind, grid in grids.items():
        best, selected, trials = None, None, []
        for params in grid:
            model = ClassicalModel(kind, seed=settings["selection_seed"], **params).fit(raw[indices], labels, groups)
            p, _ = model.predict(val_raw)
            score = criterion(data, p)
            trials.append({"settings": params, "reference_edtc": score[0], "brier": -score[1]})
            print(f"{kind} {params}: EDTC={score[0]:.6f}, Brier={-score[1]:.6f}", flush=True)
            if best is None or score > best:
                best, selected = score, model
        p, u = selected.predict(val_raw)
        threshold = select_threshold(data, p)
        selected.save(run.artifact_dir("models") / f"{kind}.joblib")
        report["classical"][kind] = {"trials": trials, "settings": selected.settings,
            "threshold": float(threshold), "validation": evaluate(data, p, u, threshold=threshold)}
    for kind in ("mlp", "vqc"):
        report["loss_pilot"][kind] = {}
        for loss in ("cross_entropy", "squared_error"):
            print(f"Shared loss pilot: {kind}, {loss}, 20000 training / full clean validation", flush=True)
            torch.manual_seed(settings["selection_seed"])
            model = NeuralFusion(kind, widths=(16,), depth=2)
            try:
                info = train_neural(model, normalized[indices], labels, val_normalized, data,
                    epochs=settings["max_epochs"], patience=settings["early_stopping_patience"],
                    learning_rate=0.001, seed=settings["selection_seed"], loss_kind=loss)
                p, u = model.predict(val_normalized)
                score = criterion(data, p)
                model.save(run.artifact_dir("models") / f"{kind}_{loss}.pt")
                result = {"reference_edtc": score[0], "brier": -score[1], "training": info,
                          "validation": evaluate(data, p, u), "nonfinite": False}
                print(f"Completed {kind}/{loss}: EDTC={score[0]:.6f}, Brier={-score[1]:.6f}, epochs={info['epochs']}", flush=True)
            except FloatingPointError as error:
                result = {"nonfinite": True, "error": str(error)}
            report["loss_pilot"][kind][loss] = result
            atomic_write_json(run.artifact_dir("metrics") / "progress.json", report)
    report["shared_loss"] = choose_loss(report["loss_pilot"])
    dbf = []
    for exponent in protocol["models"]["dbf_lambda"]:
        p, _ = predict_reference(validation, "dbf", exponent=exponent)
        c = criterion(data, p)
        dbf.append({"exponent": exponent, "reference_edtc": c[0], "brier": -c[1]})
    report["dbf_trials"] = dbf
    report["dbf_exponent"] = max(dbf, key=lambda r: (r["reference_edtc"], -r["brier"], r["exponent"] == 1))["exponent"]
    report["seconds"] = time.perf_counter() - started
    atomic_write_json(run.artifact_dir("metrics") / "selection.json", report)
    required = [p for parent in ("models", "metrics", "manifests") for p in run.artifact_dir(parent).rglob("*") if p.is_file()]
    run.mark_complete(required)
    print(f"CLEAN FUSION SELECTION PASSED: loss={report['shared_loss']}, {run.root}", flush=True)


if __name__ == "__main__":
    main()
