#!/usr/bin/env python3
"""CPU-only objective comparison; no images, official validation or conflict scoring."""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from src.thesis_experiments import opinion_diagnostics as diagnostic
from src.thesis_experiments import opinion_objectives as study
from src.thesis_experiments.detector_cache import immutable_json
from src.thesis_experiments.detector_contract import load_phase2_manifest
from src.thesis_experiments.provenance import sha256_file, _git


def run(args):
    import torch
    started = time.perf_counter()
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        raise RuntimeError("this diagnostic must run without a GPU")
    if _git(REPO, "status", "--porcelain") or (args.dataset_root / "test").exists():
        raise RuntimeError("requires clean sources and locked test split")
    spec_path = REPO / "experiments/protocol/opinion_objectives_v1.json"
    spec = json.loads(spec_path.read_text())
    if sha256_file(args.reference_report) != spec["reference_report_sha256"]:
        raise ValueError("wrong training feature reference")
    phase2 = sha256_file(args.phase2_manifest)
    load_phase2_manifest(args.phase2_manifest, repository=REPO)
    reference = json.loads(args.reference_report.read_text())
    if reference["phase2_sha256"] != phase2:
        raise ValueError("detector identity mismatch")
    # Whitelist training provenance/support only; do not use any validation outcomes.
    training = {sensor: {"features": data["features"]["train"],
                        "population": data["population_support"]["train"]}
                for sensor, data in reference["modalities"].items()}
    del reference
    head_spec_path = REPO / "experiments/protocol/presence_opinions_v1.json"
    head_spec = json.loads(head_spec_path.read_text())
    if sha256_file(head_spec_path) != spec["head_specification_sha256"]:
        raise ValueError("head architecture/schedule changed")
    sources = {str(path.relative_to(REPO)): sha256_file(path) for path in (
        Path(__file__).resolve(), REPO / "src/thesis_experiments/opinion_objectives.py",
        REPO / "src/thesis_experiments/opinion_diagnostics.py", REPO / "src/thesis_experiments/presence_opinions.py",
        spec_path, head_spec_path)}
    args.output_root.mkdir(parents=True, exist_ok=False)
    report = {"status": "diagnostic_only", "primary_ready": False,
              "git_commit": _git(REPO, "rev-parse", "HEAD"), "source_sha256": sources,
              "phase2_sha256": phase2, "training_reference_sha256": spec["reference_report_sha256"],
              "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
              "official_validation_features_read": False, "images_read": False,
              "selection_policy": "report all seeds/objectives; no checkpoint, best-seed or conflict selection",
              "modalities": {}}
    for sensor in ("rgb", "thermal"):
        values, rows = study.read_training_features(training[sensor]["features"], sensor, phase2, REPO)
        selected, split = study.training_sequence_split(training[sensor]["population"], rows, sensor,
                                                        seed=spec["split_seed"])
        result = {"partition": split, "features": training[sensor]["features"],
                  "support": {name: diagnostic.support(items, sensor) for name, items in selected.items()},
                  "runs": []}
        for seed in spec["head_seeds"]:
            for objective in spec["objectives"]:
                metrics, checkpoint = study.fit_objective(values, rows, selected, split, sensor,
                    {**head_spec, "seed": seed}, objective)
                metrics["seed"] = seed
                checkpoint.update({"source_sha256": sources, "phase2_sha256": phase2,
                                   "training_feature_record": training[sensor]["features"],
                                   "partition": split})
                path = args.output_root / f"{sensor}_{objective}_seed{seed}.pt"
                torch.save(checkpoint, path)
                metrics["checkpoint_sha256"] = sha256_file(path)
                result["runs"].append(metrics)
                heldout = metrics["heldout_train"]
                print(f"{sensor}/{objective}/seed{seed}: heldout background recall={heldout['background_recall']:.4f}, "
                      f"UAV recall={heldout['uav_recall']:.4f}, Brier={heldout['population_weighted_brier']:.6f}", flush=True)
        result["summary"] = {}
        for objective in spec["objectives"]:
            runs = [run["heldout_train"] for run in result["runs"] if run["objective"] == objective]
            result["summary"][objective] = {key: {"median": float(np.median([run[key] for run in runs])),
                "min": float(min(run[key] for run in runs)), "max": float(max(run[key] for run in runs))}
                for key in ("uav_recall", "background_recall", "balanced_accuracy",
                            "population_weighted_brier", "population_weighted_log_loss")}
        report["modalities"][sensor] = result
    for relative, expected in sources.items():
        if sha256_file(REPO / relative) != expected:
            raise RuntimeError("comparison source changed during run")
    report["wall_seconds"] = time.perf_counter() - started
    immutable_json(args.output_root / "report.json", report)
    immutable_json(args.output_root / "COMPLETED.json", {"report_sha256": sha256_file(args.output_root / "report.json")})
    print("TRAINING-ONLY OBJECTIVE COMPARISON PASSED; SCIENTIFIC RESULTS REQUIRE REVIEW")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("phase2-manifest", "dataset-root", "reference-report", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
