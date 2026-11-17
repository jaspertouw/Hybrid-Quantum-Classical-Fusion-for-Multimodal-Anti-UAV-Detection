#!/usr/bin/env python3
"""Close local correctness gates before further original-experiment selection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

from src.thesis_experiments.detector_cache import iter_cache, digest
from src.thesis_experiments.features import fit_features, training_sample, transform_features
from src.thesis_experiments.io import atomic_write_bytes, atomic_write_json
from src.thesis_experiments.metrics import evaluate, evaluation_data, oracle_acceptance, robustness_summary
from src.thesis_experiments.models import ClassicalModel
from src.thesis_experiments.provenance import sha256_file
from src.thesis_experiments.verification import synthetic_run, verify_quantum_reference


def support(rows, indices):
    selected = [rows[i] for i in indices]
    negative = Counter(r["sequence"] for r in selected if not r["paired_presence"])
    return {"frames": len(selected), "absent": sum(negative.values()),
            "present": sum(r["paired_presence"] for r in selected),
            "absent_sequences": dict(sorted(negative.items())),
            "largest_absent_sequence_share": max(negative.values()) / sum(negative.values()) if negative else None,
            "keys_sha256": digest(sorted(r["key"] for r in selected))}


def training_support_audit(rows):
    if any(r["split"] != "train" for r in rows):
        raise ValueError("support assessment uses original training only")
    strata = defaultdict(list)
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[row["sequence"]].append(row)
    for sequence, group in by_sequence.items():
        strata[any(not r["paired_presence"] for r in group)].append(sequence)
    heldout = set()
    for sequences in strata.values():
        ranked = sorted(sequences, key=lambda s: hashlib.sha256(f"sample-support-v1:11:{s}".encode()).digest())
        heldout.update(ranked[::5])
    fit = [r for r in rows if r["sequence"] not in heldout]
    evaluation = [r for r in rows if r["sequence"] in heldout]
    if any(len({r["paired_presence"] for r in part}) != 2 for part in (fit, evaluation)):
        raise ValueError("fixed training-only support split lacks a class; do not retry split")
    transforms = fit_features(fit)
    raw, _, _ = transform_features(fit, transforms)
    validation_raw, _, _ = transform_features(evaluation, transforms)
    y = np.array([r["paired_presence"] for r in fit])
    groups = np.array([r["sequence"] for r in fit])
    data = evaluation_data(evaluation)
    result = {"scope": "Training-only selector holdout; frozen detectors previously saw these sequences",
              "split_rule": "SHA256 rank per any-background stratum, every fifth sequence held out; seed 11; no retries",
              "heldout_sequences": sorted(heldout), "heldout_frames": len(evaluation),
              "heldout_absent": int((~data["label"]).sum()), "full_training_caps": {}, "learning_curve": {}}
    for cap in (20000, 40000, len(rows)):
        result["full_training_caps"][str(cap)] = support(rows, training_sample(rows, cap))
    for cap in (20000, 40000, len(fit)):
        indices = training_sample(fit, cap)
        models = {}
        for kind, settings in (("svm", {"C": 1., "gamma": 0.1}), ("forest", {"max_depth": 8, "min_samples_leaf": 5})):
            print(f"Training-only support check: {kind}, cap={cap}", flush=True)
            try:
                model = ClassicalModel(kind, **settings).fit(raw[indices], y[indices], groups[indices])
                p, u = model.predict(validation_raw)
                metrics = evaluate(data, p, u)
                models[kind] = {"edtc": metrics["edtc_sequence_mean"], "brier": metrics["brier"],
                    "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
                    "brier_by_class": metrics["brier_support"], "fold_support": getattr(model, "fold_support", None)}
            except ValueError as error:
                models[kind] = {"unsupported": str(error)}
        result["learning_curve"][str(cap)] = {"sample": support(fit, indices), "models": models}
    result["interpretation"] = "More frames do not create independent background sequences. This diagnostic cannot prove minority support sufficient or automatically freeze a cap; assess recall and classwise Brier jointly. Existing choices remain provisional."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/verification/fusion-local-v1"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/caches/v1"))
    parser.add_argument("--pilot-run", type=Path, default=Path("outputs/successor/protocol-1.0.0/runs/original-fusion-pilot-v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output / "PARTIAL.json", {"status": "running"})
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        first, p1 = synthetic_run(args.output / "synthetic_first")
        second, p2 = synthetic_run(args.output / "synthetic_repeat")
        if first != second:
            raise AssertionError("fresh synthetic cache-to-metrics runs differ")
        differences = {name: float(np.max(np.abs(p1[name] - p2[name]))) for name in p1}
        if max(differences.values()) > 1e-12:
            raise AssertionError("fresh fitted predictions differ")
        dense = verify_quantum_reference()
        print("Synthetic repeat and independent quantum checks passed", flush=True)
        training = list(iter_cache(args.cache_root / "clean/train"))
        support_report = training_support_audit(training)
        del training
    validation = list(iter_cache(args.cache_root / "clean/val"))
    bound = oracle_acceptance(evaluation_data(validation))
    if abs(bound["edtc_sequence_mean"] - 0.8093020270444736) > 1e-10:
        raise AssertionError("fixed-candidate oracle differs from original cache audit")
    previous = json.loads((args.pilot_run / "metrics/report.json").read_text())
    metadata = json.loads((args.cache_root / "pilot/val_3_per_sequence/CACHE.json").read_text())
    conditions = [c["id"] for c in metadata["context"]["conditions"] if c["corruption"] not in ("clean", "dropout")]
    robustness = {name: robustness_summary(result["pilot_conditions"], conditions) for name, result in previous["models"].items()}
    inherited = any(name.startswith("lib.models.yolo_evidential") for name in sys.modules)
    if inherited:
        raise AssertionError("cache-only verification imported inherited detector code")
    report = {"test_accessed": False, "synthetic_repeat": {"identical_metrics_and_histories": True,
              "prediction_max_absolute_difference": differences, "models": list(p1)},
              "independent_quantum_reference": dense, "training_support": support_report,
              "fixed_box_oracle": bound, "robustness": robustness,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                  "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                  "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                  "simulator": "Torch exact statevector complex128", "inherited_detector_imported": inherited},
              "source_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in
                  [Path(__file__), *sorted((REPO / "src/thesis_experiments").glob("*.py"))]}}
    json.dumps(report, allow_nan=False)
    atomic_write_json(args.output / "report.json", report)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    atomic_write_bytes(args.output / "environment.txt", freeze.stdout.encode())
    files = [p for p in args.output.rglob("*") if p.is_file() and p.name not in ("PARTIAL.json", "COMPLETED.json")]
    atomic_write_json(args.output / "COMPLETED.json", {"files": {str(p.relative_to(args.output)): sha256_file(p) for p in files}})
    (args.output / "PARTIAL.json").unlink()
    print(f"LOCAL FUSION VERIFICATION PASSED: {args.output}", flush=True)


if __name__ == "__main__":
    main()
