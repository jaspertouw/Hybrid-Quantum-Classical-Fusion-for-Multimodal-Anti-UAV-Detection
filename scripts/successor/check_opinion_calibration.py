#!/usr/bin/env python3
"""Three-way training-sequence calibration diagnostic on cached features, CPU only."""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from src.thesis_experiments import opinion_calibration as calibration
from src.thesis_experiments import opinion_diagnostics as diagnostic
from src.thesis_experiments import opinion_objectives as study
from src.thesis_experiments import presence_opinions as opinions
from src.thesis_experiments.detector_cache import immutable_json
from src.thesis_experiments.detector_contract import load_phase2_manifest
from src.thesis_experiments.provenance import sha256_file, _git


def run(args):
    import torch
    start = time.perf_counter()
    torch.set_num_threads(8)
    if torch.cuda.is_available() or _git(REPO, "status", "--porcelain") or (args.dataset_root / "test").exists():
        raise RuntimeError("requires CPU, clean sources and locked test split")
    spec_path = REPO / "experiments/protocol/opinion_calibration_v1.json"
    spec = json.loads(spec_path.read_text())
    if sha256_file(args.objective_report) != spec["objective_report_sha256"] or sha256_file(args.feature_reference) != spec["feature_reference_sha256"]:
        raise ValueError("wrong frozen diagnostic references")
    phase2 = sha256_file(args.phase2_manifest)
    load_phase2_manifest(args.phase2_manifest, repository=REPO)
    objective = json.loads(args.objective_report.read_text())
    feature_reference = json.loads(args.feature_reference.read_text())
    if objective["phase2_sha256"] != phase2 or feature_reference["phase2_sha256"] != phase2:
        raise ValueError("detector identity changed")
    training_population = {sensor: data["population_support"]["train"]
                           for sensor, data in feature_reference["modalities"].items()}
    del feature_reference
    for relative, expected in objective["source_sha256"].items():
        if sha256_file(REPO / relative) != expected:
            raise ValueError(f"reference head source changed: {relative}")
    sources = {**objective["source_sha256"], **{str(path.relative_to(REPO)): sha256_file(path) for path in (
        Path(__file__).resolve(), REPO / "src/thesis_experiments/opinion_calibration.py", spec_path)}
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    report = {"status": "diagnostic_only", "primary_ready": False, "phase2_sha256": phase2,
              "source_sha256": sources, "git_commit": _git(REPO, "rev-parse", "HEAD"),
              "objective_report_sha256": spec["objective_report_sha256"],
              "feature_reference_sha256": spec["feature_reference_sha256"],
              "images_read": False, "official_validation_features_read": False,
              "evidential_uncertainty_calibration_claimed": False, "modalities": {}}
    for sensor in ("rgb", "thermal"):
        parent = objective["modalities"][sensor]
        if {(run["objective"], run["seed"]) for run in parent["runs"]} != {
                (name, seed) for name in ("natural_prevalence", "class_balanced") for seed in (11, 22, 33)}:
            raise ValueError("expected every predeclared objective/seed")
        values, rows = study.read_training_features(parent["features"], sensor, phase2, REPO)
        selected, partition = calibration.calibration_partition(training_population[sensor], rows,
            parent["partition"], sensor, seed=spec["partition_seed"])
        pi_cal, pi_eval = (partition["population_prevalence"][name] for name in ("calibration", "evaluation"))
        result = {"partition": partition, "features": parent["features"],
                  "support": {name: diagnostic.support(items, sensor) for name, items in selected.items()},
                  "runs": []}
        for previous in parent["runs"]:
            name, seed = previous["objective"], previous["seed"]
            checkpoint_path = args.objective_report.parent / f"{sensor}_{name}_seed{seed}.pt"
            if sha256_file(checkpoint_path) != previous["checkpoint_sha256"]:
                raise ValueError("head checkpoint hash mismatch")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint["phase2_sha256"] != phase2 or set(checkpoint["train_keys"]) != {row["key"] for row in selected["head_fit"]}:
                raise ValueError("head-fit membership mismatch")
            head = opinions.build_head(checkpoint["input_size"], checkpoint["specification"]["hidden_size"])
            head.load_state_dict(checkpoint["state_dict"])
            head.eval()
            head.requires_grad_(False)
            predicted = {}
            for part in ("calibration", "evaluation"):
                x = diagnostic.subset_features(values, rows, selected[part])
                x = ((x - np.array(checkpoint["mean"])) / np.array(checkpoint["scale"])).astype(np.float32)
                with torch.no_grad():
                    predicted[part] = head(torch.from_numpy(x)).numpy()
            fitted = calibration.fit_scaling(predicted["calibration"],
                [row[sensor]["present"] for row in selected["calibration"]], pi_cal,
                regularization=spec["regularization"], bound=spec["log_scale_bound"], max_iter=spec["max_iter"])
            metrics = {}
            for part, pi in (("calibration", pi_cal), ("evaluation", pi_eval)):
                raw = predicted[part]
                scaled = calibration.apply_scaling(raw, fitted["log_scales"], bound=spec["log_scale_bound"])
                metrics[part] = {"raw": study.probability_report(raw, selected[part], sensor, pi, pi_cal),
                                 "calibrated": study.probability_report(scaled, selected[part], sensor, pi, pi_cal)}
            passed = calibration.joint_gate(metrics["evaluation"]["raw"], metrics["evaluation"]["calibrated"])
            run = {"objective": name, "seed": seed, "calibrator": fitted, "metrics": metrics,
                   "joint_gate": passed, "head_sha256": previous["checkpoint_sha256"],
                   "evaluation_per_sequence": diagnostic.per_sequence_report(
                       calibration.apply_scaling(predicted["evaluation"], fitted["log_scales"]), selected["evaluation"], sensor)}
            result["runs"].append(run)
            immutable_json(args.output_root / f"{sensor}_{name}_seed{seed}_calibration.json",
                {"status": "diagnostic_only", "head_sha256": run["head_sha256"], "calibrator": fitted,
                 "fit_keys": [row["key"] for row in selected["calibration"]],
                 "evaluation_keys": [row["key"] for row in selected["evaluation"]], "source_sha256": sources})
            eval_metrics = metrics["evaluation"]
            print(f"{sensor}/{name}/seed{seed}: background recall "
                  f"{eval_metrics['raw']['background_recall']:.4f}->{eval_metrics['calibrated']['background_recall']:.4f}, "
                  f"Brier {eval_metrics['raw']['population_weighted_brier']:.6f}->"
                  f"{eval_metrics['calibrated']['population_weighted_brier']:.6f}, joint_gate={passed}", flush=True)
        result["all_seeds_gate"] = {name: all(run["joint_gate"] for run in result["runs"] if run["objective"] == name)
                                    for name in ("natural_prevalence", "class_balanced")}
        report["modalities"][sensor] = result
    for relative, expected in sources.items():
        if sha256_file(REPO / relative) != expected:
            raise RuntimeError("calibration source changed during run")
    if sha256_file(args.objective_report) != spec["objective_report_sha256"] or sha256_file(args.feature_reference) != spec["feature_reference_sha256"]:
        raise RuntimeError("reference artifact changed during run")
    report["wall_seconds"] = time.perf_counter() - start
    immutable_json(args.output_root / "report.json", report)
    immutable_json(args.output_root / "COMPLETED.json", {"report_sha256": sha256_file(args.output_root / "report.json")})
    print("TRAINING-ONLY CALIBRATION DIAGNOSTIC PASSED; SCIENTIFIC GATES REQUIRE REVIEW")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("phase2-manifest", "dataset-root", "objective-report", "feature-reference", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
