#!/usr/bin/env python3
"""Compare small and expanded clean training samples; no conflict-based tuning."""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import cv2
import numpy as np

from src.thesis_experiments import presence_opinions as opinions
from src.thesis_experiments import opinion_diagnostics as diagnostic
from src.thesis_experiments.detector_cache import immutable_json
from src.thesis_experiments.provenance import sha256_file, _git
from scripts.successor.build_detector_cache import contracts


def run(args):
    import torch
    from src.thesis_experiments.inference_runtime import configure_inference
    from src.thesis_experiments.degraded_inputs import prepare_conditioned_sensor
    from src.thesis_experiments.corruptions import Condition, FrameKey
    for path in (REPO / "resources/EDTC", REPO / "resources/EDTC/yolov5"):
        sys.path.insert(0, str(path))
    from lib.models.yolo_evidential.model import EvidentialYOLO

    started = time.perf_counter()
    if _git(REPO, "status", "--porcelain"):
        raise RuntimeError("diagnostic requires committed, clean sources")
    _, pair, phase5_path = contracts(args, REPO)
    spec_path = REPO / "experiments/protocol/presence_diagnostic_v1.json"
    spec = json.loads(spec_path.read_text())
    pilot_spec_path = REPO / "experiments/protocol/presence_opinions_v1.json"
    pilot_spec = json.loads(pilot_spec_path.read_text())
    reference_bytes = args.reference_report.read_bytes()
    reference = json.loads(reference_bytes)
    if (reference["git_commit"] != spec["reference_commit"]
            or reference["phase2_sha256"] != sha256_file(args.phase2_manifest)
            or spec["pilot_specification_sha256"] != sha256_file(pilot_spec_path)
            or reference["source_sha256"]["experiments/protocol/presence_opinions_v1.json"] != sha256_file(pilot_spec_path)):
        raise ValueError("diagnostic/reference specification mismatch")
    args.output_root.mkdir(parents=True, exist_ok=False)
    configure_inference()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    rows = [json.loads(line) for line in pair.development_manifest_path.read_text().splitlines()]
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    sources = {str(path.relative_to(REPO)): sha256_file(path) for path in
        (Path(__file__).resolve(), REPO / "src/thesis_experiments/opinion_diagnostics.py",
         REPO / "src/thesis_experiments/presence_opinions.py", spec_path)}
    report = {"status": "diagnostic_only", "primary_ready": False,
              "git_commit": _git(REPO, "rev-parse", "HEAD"), "source_sha256": sources,
              "reference_report_sha256": sha256_file(args.reference_report),
              "phase2_sha256": sha256_file(args.phase2_manifest),
              "phase5_sha256": sha256_file(phase5_path), "modalities": {},
              "policy": "No corrupted images or conflict objective; fixed epochs and same validation keys; descriptive correlated-frame metrics"}
    for sensor in ("rgb", "thermal"):
        small = opinions.sample_by_class(train, sensor, 128, split="train", seed=pilot_spec["seed"])
        expanded = opinions.sample_by_class(train, sensor, spec["expanded_per_class"], split="train", seed=pilot_spec["seed"])
        validation = opinions.sample_by_class(val, sensor, 128, split="val", seed=pilot_spec["seed"])
        if ([row["key"] for row in small] != reference["modalities"][sensor]["train_keys"]
                or [row["key"] for row in validation] != reference["modalities"][sensor]["validation_keys"]):
            raise ValueError("fixed pilot sample membership changed")
        identities = {"phase2_sha256": report["phase2_sha256"], "phase5_sha256": report["phase5_sha256"],
                      "source_sha256": sources, "torch": torch.__version__, "numpy": np.__version__,
                      "opencv": cv2.__version__, "gpu": torch.cuda.get_device_name(0),
                      "precision": "strict-fp32", "selected_checkpoint_sha256": sha256_file(pair.selected_checkpoints[sensor])}
        detector, extractor = None, None

        def extract(frames):
            nonlocal detector, extractor
            if detector is None:
                detector = EvidentialYOLO.load(str(pair.selected_checkpoints[sensor]),
                    str(pair.clean_checkpoints[sensor]), device=torch.device("cuda:0"), K=2).eval()
                extractor = opinions.FrozenFrameFeatures(detector)
            stride = int(detector.stride.max())
            output = []
            for offset in range(0, len(frames), 32):
                chunk = frames[offset:offset + 32]
                groups = {}
                for index, row in enumerate(chunk):
                    path = (args.derived_root / row[sensor]["image"]).resolve()
                    if not path.is_relative_to(args.derived_root.resolve()):
                        raise ValueError("image outside derived root")
                    prepared = prepare_conditioned_sensor(path, sensor,
                        FrameKey(row["split"], row["sequence"], row["frame"]), Condition(), stride=stride)
                    groups.setdefault(prepared.pixels_chw.shape, []).append((index, prepared.pixels_chw))
                result = [None] * len(chunk)
                for group in groups.values():
                    images = torch.from_numpy(np.stack([item[1] for item in group])).cuda()
                    features = extractor(images)
                    if offset == 0 and not torch.allclose(features[:1], extractor(images[:1]), rtol=1e-4, atol=1e-5):
                        raise RuntimeError("pooled feature single/batch mismatch")
                    for (index, _), value in zip(group, features.cpu().numpy()):
                        result[index] = value
                output.extend(result)
                if offset % 512 == 0:
                    print(f"{sensor}: extracted {min(offset + 32, len(frames))}/{len(frames)} clean frames", flush=True)
            return np.stack(output)

        train_features, train_record = diagnostic.cached_features(args.feature_root,
            diagnostic.feature_context(expanded, sensor, identities), lambda: extract(expanded))
        val_features, val_record = diagnostic.cached_features(args.feature_root,
            diagnostic.feature_context(validation, sensor, identities), lambda: extract(validation))
        if extractor is not None:
            extractor.close()
        del extractor, detector
        torch.cuda.empty_cache()
        prevalence = sum(row[sensor]["present"] for row in train) / len(train)
        val_prevalence = sum(row[sensor]["present"] for row in val) / len(val)
        result = {"training_prevalence": prevalence, "validation_prevalence": val_prevalence,
                  "population_support": {"train": diagnostic.support(train, sensor), "val": diagnostic.support(val, sensor)},
                  "sample_support": {"small": diagnostic.support(small, sensor),
                      "expanded": diagnostic.support(expanded, sensor), "validation": diagnostic.support(validation, sensor)},
                  "features": {"train": train_record, "validation": val_record}, "arms": {}}
        for name, sample in (("small", small), ("expanded", expanded)):
            metrics, checkpoint, evidence = diagnostic.train_arm(
                diagnostic.subset_features(train_features, expanded, sample), sample,
                val_features, validation, sensor, pilot_spec, prevalence, val_prevalence)
            checkpoint.update({"source_sha256": sources, "phase2_sha256": report["phase2_sha256"],
                               "feature_records": result["features"]})
            checkpoint_path = args.output_root / f"{sensor}_{name}.pt"
            torch.save(checkpoint, checkpoint_path)
            metrics["checkpoint_sha256"] = sha256_file(checkpoint_path)
            result["arms"][name] = metrics
            probabilities = opinions.opinion(evidence)["probabilities"][:, 0]
            with (args.output_root / f"{sensor}_{name}_validation.jsonl").open("w") as handle:
                for row, probability in zip(validation, probabilities):
                    handle.write(json.dumps({"key": row["key"], "sequence": row["sequence"],
                        "present": row[sensor]["present"], "p_uav": float(probability)}) + "\n")
            print(f"{sensor}/{name}: train background recall={metrics['train']['background_recall']:.4f}, "
                  f"val background recall={metrics['validation']['background_recall']:.4f}, "
                  f"val Brier={metrics['validation']['population_weighted_brier']:.6f}, "
                  f"gate={metrics['validation']['both_class_learning_gate']}", flush=True)
        report["modalities"][sensor] = result
    contracts(args, REPO)
    for relative, expected in sources.items():
        if sha256_file(REPO / relative) != expected:
            raise RuntimeError("diagnostic sources changed during run")
    if args.reference_report.read_bytes() != reference_bytes:
        raise RuntimeError("reference changed during run")
    report["wall_seconds"] = time.perf_counter() - started
    immutable_json(args.output_root / "report.json", report)
    immutable_json(args.output_root / "COMPLETED.json", {"report_sha256": sha256_file(args.output_root / "report.json")})
    print("CLEAN OPINION DIAGNOSTIC EXECUTION PASSED; PRIMARY REMAINS PAUSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("phase2-manifest", "dataset-root", "derived-root", "reference-report", "feature-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
