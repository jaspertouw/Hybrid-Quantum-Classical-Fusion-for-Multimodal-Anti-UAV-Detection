#!/usr/bin/env python3
"""Resume the original presence-fusion experiment using verified development caches."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import numpy as np
import torch

from src.thesis_experiments.config import load_experiment_config
from src.thesis_experiments.detector_cache import iter_cache
from src.thesis_experiments.detector_contract import EXPECTED_PROTOCOL_SHA256
from src.thesis_experiments.features import fit_features, transform_features, training_sample, validation_sample
from src.thesis_experiments.fusion import disagreement, predict_reference
from src.thesis_experiments.io import atomic_write_bytes, atomic_write_json
from src.thesis_experiments.metrics import evaluate, evaluation_data, paired_bootstrap, select_threshold, sequence_scores
from src.thesis_experiments.models import ClassicalModel
from src.thesis_experiments.neural_fusion import NeuralFusion, train_neural
from src.thesis_experiments.provenance import sha256_file
from src.thesis_experiments.run import RunLayout


def read_cache(path, split, protocol):
    rows = list(iter_cache(path))
    sequences = sorted({r["sequence"] for r in rows})
    membership = hashlib.sha256("".join(s + "\n" for s in sequences).encode()).hexdigest()
    if (any(r["split"] != split for r in rows) or membership != protocol["dataset"]["membership_sha256"][split]):
        raise ValueError(f"unexpected {split} cache membership")
    metadata = json.loads((path / "CACHE.json").read_text())
    return rows, {"root": str(path.resolve()), "context_id": metadata["context_id"],
                  "context": metadata["context"], "completion_sha256": sha256_file(path / "COMPLETED.json")}


def report_text(report):
    lines = ["# Original presence-fusion development pilot", "", report["pilot_settings"]["purpose"],
             "No test evaluation, detector retraining, candidate switching or final hyperparameter/loss selection.",
             "Classical models use up to 20,000 clean training pairs; neural runs use 2,048 pairs and five epochs for runtime verification.", "",
             "| Model | Clean validation EDTC | Brier | Selected development threshold | Pilot mean EDTC (30 corruptions) |",
             "|---|---:|---:|---:|---:|"]
    for name, value in report["models"].items():
        clean = value["full_validation"]
        conditions = value["pilot_conditions"]
        corruption = [v["edtc_sequence_mean"] for k, v in conditions.items()
                      if k != "clean" and "dropout" not in k and "edtc_sequence_mean" in v]
        edtc = f"{clean['edtc_sequence_mean']:.6f}" if "edtc_sequence_mean" in clean else "—"
        robust = f"{np.mean(corruption):.6f}" if corruption else "—"
        lines.append(f"| {name} | {edtc} | {clean['brier']:.6f} | {value.get('threshold', '—')} | {robust} |")
    lines += ["", report["pilot_settings"]["limitation"],
              "Pilot corruption means cover only 201 pairs per condition and do not replace full validation/test robustness evaluation.",
              "SVM sigmoid calibration uses out-of-fold sequence-grouped training predictions; all feature caps/min-max statistics are training-only.",
              "The VQC uses exact four-qubit statevector simulation and a classical evidential readout; its uncertainty does not originate in the circuit.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/caches/v1"))
    parser.add_argument("--settings", type=Path, default=Path("experiments/config/fusion_pilot.json"))
    parser.add_argument("--run-id", default="original-fusion-pilot-v1")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    config = load_experiment_config(output_root=args.output_root)
    if sha256_file(config.protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("primary protocol identity changed")
    protocol = json.loads(config.protocol_path.read_text())
    run = RunLayout(config, args.run_id)
    if run.root.exists():
        raise FileExistsError(f"choose a new run ID: {run.root}")
    source = [REPOSITORY / "src/thesis_experiments" / name for name in (
        "features.py", "fusion.py", "metrics.py", "models.py", "neural_fusion.py")]
    run.initialize([args.settings, Path(__file__), *source])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    print("Verifying and reading development caches", flush=True)
    train, tr_record = read_cache(args.cache_root / "clean/train", "train", protocol)
    validation, va_record = read_cache(args.cache_root / "clean/val", "val", protocol)
    pilot, pi_record = read_cache(args.cache_root / "pilot/val_3_per_sequence", "val", protocol)
    for record in (va_record, pi_record):
        for field in ("phase2_manifest_sha256", "development_manifest_sha256", "corruption_specification_sha256"):
            if record["context"][field] != tr_record["context"][field]:
                raise ValueError(f"cache contracts differ: {field}")
    if set(r["sequence"] for r in train) & set(r["sequence"] for r in validation):
        raise ValueError("training and validation sequences overlap")
    fitted = fit_features(train)
    atomic_write_json(run.artifact_dir("models") / "features.json", fitted)
    raw_train, norm_train, train_clip = transform_features(train, fitted)
    raw_val, norm_val, val_clip = transform_features(validation, fitted)
    raw_pilot, norm_pilot, pilot_clip = transform_features(pilot, fitted)
    y = np.array([r["paired_presence"] for r in train], dtype=bool)
    groups = np.array([r["sequence"] for r in train])
    sample = training_sample(train, settings["classical_training_cap"], settings["seed"])
    neural_sample = training_sample(train, settings["neural_training_cap"], settings["seed"])
    val_sample = validation_sample(validation, settings["neural_validation_cap"], settings["seed"])
    atomic_write_json(run.artifact_dir("manifests") / "samples.json", {
        "classical_training": [train[i]["key"] for i in sample],
        "neural_training": [train[i]["key"] for i in neural_sample],
        "neural_validation": [validation[i]["key"] for i in val_sample],
        "classical_present": int(y[sample].sum()), "classical_absent": int((~y[sample]).sum()),
        "neural_present": int(y[neural_sample].sum()), "neural_absent": int((~y[neural_sample]).sum())})
    shared_data = evaluation_data(validation)
    pilot_indices = defaultdict(list)
    for i, row in enumerate(pilot):
        pilot_indices[row["condition"]].append(i)
    report = {"pilot_settings": settings, "input_caches": {"train": tr_record, "validation": va_record, "pilot": pi_record},
              "preprocessing": {"train": train_clip, "validation": val_clip, "pilot": pilot_clip},
              "test_accessed": False, "hardware": platform.platform(), "numpy": np.__version__,
              "torch": torch.__version__, "models": {}, "disagreement": {}}
    for name, rows in (("full_validation", validation), ("pilot", pilot)):
        d = disagreement(rows)
        report["disagreement"][name] = {key: {"mean": float(v.mean()), "max": float(v.max())} for key, v in d.items()}

    def evaluate_model(name, predict, *, sensor=None, box_scores=True, fitted_info=None):
        print(f"Evaluating {name}", flush=True)
        data = evaluation_data(validation, sensor) if sensor else shared_data
        p, u = predict(validation, raw_val, norm_val)
        threshold = select_threshold(data, p) if box_scores else None
        result = {"full_validation": evaluate(data, p, u, threshold=threshold or 0.5, box_scores=box_scores),
                  "pilot_conditions": {}, "fit": fitted_info}
        if threshold is not None:
            result["threshold"] = float(threshold)
            result["reference_threshold_edtc"] = float(evaluate(data, p, u)["edtc_sequence_mean"])
        np.savez_compressed(run.artifact_dir("caches") / f"{name}_validation.npz", probability=p,
                            uncertainty=np.array([]) if u is None else u)
        for condition, indices in sorted(pilot_indices.items()):
            rows = [pilot[i] for i in indices]
            cp, cu = predict(rows, raw_pilot[indices], norm_pilot[indices])
            result["pilot_conditions"][condition] = evaluate(evaluation_data(rows, sensor), cp, cu,
                threshold=threshold or 0.5, box_scores=box_scores)
        result["natural_clean"] = {}
        for tag in ("LI", "OC", "FM", "TC"):
            indices = [i for i, row in enumerate(validation) if tag in row["attributes"]]
            if indices:
                result["natural_clean"][tag] = evaluate(evaluation_data([validation[i] for i in indices], sensor),
                    p[indices], None if u is None else u[indices], threshold=threshold or 0.5, box_scores=box_scores)
        warm_rows = validation[:128]
        for _ in range(settings["timing_warmup"]):
            predict(warm_rows, raw_val[:128], norm_val[:128])
        before = time.perf_counter()
        for _ in range(settings["timing_repetitions"]):
            predict(warm_rows, raw_val[:128], norm_val[:128])
        result["seconds_per_128_fusion_only"] = (time.perf_counter() - before) / settings["timing_repetitions"]
        report["models"][name] = result
        atomic_write_json(run.artifact_dir("metrics") / f"{name}.json", result)

    prevalence = float(y.mean())
    for method in ("always_present", "prevalence", "rgb", "thermal", "naive", "abf", "dbf"):
        evaluate_model(method, lambda rows, raw, norm, m=method: predict_reference(rows, m, prevalence=prevalence),
                       sensor=method if method in ("rgb", "thermal") else None,
                       box_scores=method not in ("always_present", "prevalence"))
    for kind in ("svm", "forest"):
        print(f"Fitting fixed {kind} configuration on {len(sample)} training pairs", flush=True)
        before = time.perf_counter()
        model = ClassicalModel(kind, seed=settings["seed"], **settings[kind]).fit(raw_train[sample], y[sample], groups[sample])
        info = {"seconds": time.perf_counter() - before, "configuration": settings[kind],
                "training_pairs": len(sample), "folds": getattr(model, "fold_support", None)}
        model.save(run.artifact_dir("models") / f"{kind}.joblib")
        evaluate_model(kind, lambda rows, raw, norm, m=model: m.predict(raw), fitted_info=info)
    for kind in ("gate", "mlp", "vqc"):
        print(f"Training bounded {kind} runtime pilot", flush=True)
        torch.manual_seed(settings["seed"])
        model = NeuralFusion(kind, widths=settings["mlp_widths"], depth=settings["vqc_depth"])
        tx, vx = (raw_train, raw_val) if kind == "gate" else (norm_train, norm_val)
        info = train_neural(model, tx[neural_sample], y[neural_sample], vx[val_sample],
            evaluation_data([validation[i] for i in val_sample]), epochs=settings["neural_epochs"],
            learning_rate=settings["learning_rate"], seed=settings["seed"], loss_kind=settings["loss"])
        info.update({"runtime_pilot_only": True, "training_pairs": len(neural_sample), "validation_pairs": len(val_sample)})
        model.save(run.artifact_dir("models") / f"{kind}.pt")
        evaluate_model(kind, lambda rows, raw, norm, m=model: m.predict(raw if m.kind == "gate" else norm), fitted_info=info)
    dms = {sequence: float(value) for sequence, value in zip(shared_data["sequences"],
            sequence_scores(shared_data, np.ones(len(validation)))[0])}
    report["emit_dms_sequence_mean_edtc"] = float(np.mean(list(dms.values())))
    report["development_intervals_vs_emit_dms"] = {name: paired_bootstrap(value["full_validation"]["per_sequence_edtc"], dms)
        for name, value in report["models"].items() if "per_sequence_edtc" in value["full_validation"] and name not in ("rgb", "thermal")}
    report["application_seconds"] = time.perf_counter() - started
    atomic_write_json(run.artifact_dir("metrics") / "report.json", report)
    atomic_write_bytes(run.artifact_dir("metrics") / "report.md", report_text(report).encode())
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
    atomic_write_bytes(run.artifact_dir("manifests") / "environment.txt", freeze.stdout.encode())
    required = [p for parent in ("models", "metrics", "manifests", "caches") for p in run.artifact_dir(parent).rglob("*") if p.is_file()]
    run.mark_complete(required)
    print(f"ORIGINAL FUSION PILOT PASSED: {run.root}", flush=True)


if __name__ == "__main__":
    main()
