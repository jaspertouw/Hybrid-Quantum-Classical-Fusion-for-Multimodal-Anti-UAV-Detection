#!/usr/bin/env python3
"""Build clean development caches or a sampled 33-condition validation pilot."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import cv2
import numpy as np

from src.thesis_experiments import detector_cache as cache
from src.thesis_experiments.corruptions import Condition, FrameKey, conditions
from src.thesis_experiments.degraded_inputs import prepare_conditioned_sensor
from src.thesis_experiments.detector_preprocessing import prepare_image
from src.thesis_experiments.detector_contract import load_phase2_manifest, load_detector_pair
from src.thesis_experiments.provenance import sha256_file, _git


def descriptor(condition):
    return {"id": condition.identifier, "corruption": condition.corruption,
            "sensor": condition.sensor, "severity": condition.severity}


def select_frames(manifest, split, *, pilot=False, per_sequence=3, sequence_limit=None):
    grouped = defaultdict(list)
    with manifest.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split:
                grouped[row["sequence"]].append(row)
    for sequence in grouped:
        grouped[sequence].sort(key=lambda frame: frame["frame"])
    if not grouped:
        raise ValueError(f"manifest contains no {split} frames")
    if sequence_limit is not None:
        grouped = {key: grouped[key] for key in sorted(grouped)[:sequence_limit]}
    if pilot:
        for sequence, rows in grouped.items():
            count = min(per_sequence, len(rows))
            indices = ([len(rows) // 2] if count == 1 else
                       [(2 * i * (len(rows) - 1) + count - 1) // (2 * (count - 1)) for i in range(count)])
            grouped[sequence] = [rows[index] for index in indices]
    return dict(sorted(grouped.items()))


def contracts(args, repository):
    if (args.dataset_root / "test").exists():
        raise RuntimeError("development cache requires the test split to remain absent")
    phase5_path = repository / "experiments/protocol/corruptions_v1.json"
    phase5 = json.loads(phase5_path.read_text())
    if phase5["status"] != "frozen" or sha256_file(args.phase2_manifest) != phase5["phase2_completion_sha256"]:
        raise ValueError("Phase 2/5 identity mismatch")
    for relative, expected in phase5["implementation_sha256"].items():
        if sha256_file(repository / relative) != expected:
            raise RuntimeError(f"frozen corruption source changed: {relative}")
    phase2 = load_phase2_manifest(args.phase2_manifest, repository=repository)
    pair = load_detector_pair(Path(phase2["detector_pair"]["path"]))
    if tuple(json.loads(pair.protocol_path.read_text())["features"]["order"]) != cache.FEATURE_ORDER:
        raise ValueError("cache feature order differs from the frozen protocol")
    return phase2, pair, phase5_path


class Engine:
    def __init__(self, pair, derived_root, workers):
        for path in (REPOSITORY / "resources/EDTC", REPOSITORY / "resources/EDTC/yolov5"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import torch
        from lib.models.yolo_evidential.model import EvidentialYOLO
        from src.thesis_experiments.inference_runtime import configure_inference

        configure_inference()
        torch.set_num_threads(workers)
        cv2.setNumThreads(1)
        self.torch, self.root, self.workers = torch, derived_root, workers
        self.models = {}
        self.counts = Counter()
        self.repeat_checked = set()
        self.hardware = {"gpu": torch.cuda.get_device_name(0),
                         "cuda_runtime": torch.version.cuda, "precision": "strict-fp32"}
        for sensor in cache.SENSORS:
            model = EvidentialYOLO.load(str(pair.selected_checkpoints[sensor]),
                                        str(pair.clean_checkpoints[sensor]), device=torch.device("cuda:0"), K=2).eval()
            model.model[-1].use_cls_for_conf = False
            self.models[sensor] = model

    def infer(self, frames, sensor, condition):
        from lib.models.yolo_evidential.postprocess import select_successor_batch
        from scripts.successor.extract_validation_candidates import _candidate_record

        torch = self.torch
        model = self.models[sensor]
        stride = int(model.stride.max())
        output_rows = {}

        def prepare(frame):
            path = (self.root / frame[sensor]["image"]).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError("image escapes derived-data root")
            if condition.corruption == "clean":
                pixels, _, geometry = prepare_image(path, 640, stride)
                return frame, pixels, geometry, 0.0, None
            result = prepare_conditioned_sensor(path, sensor,
                FrameKey(frame["split"], frame["sequence"], frame["frame"]), condition, stride=stride)
            if result is None:
                raise ValueError("dropout must bypass Engine.infer")
            return frame, result.pixels_chw, result.geometry, result.clipped_fraction, result.noise_seed_hex

        with torch.inference_mode(), ThreadPoolExecutor(max_workers=self.workers) as executor:
            for offset in range(0, len(frames), 128):
                preparation_start = time.perf_counter()
                prepared = list(executor.map(prepare, frames[offset:offset + 128]))
                self.counts["preparation_seconds"] += time.perf_counter() - preparation_start
                groups = defaultdict(list)
                for item in prepared:
                    groups[item[1].shape].append(item)
                for group in groups.values():
                    images = torch.from_numpy(np.stack([item[1] for item in group])).to("cuda:0")
                    if images.dtype == torch.uint8:
                        images = images.float() / 255.0
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = model(images)
                    selected = select_successor_batch(output[0], output[4])
                    torch.cuda.synchronize()
                    self.counts["model_seconds"] += time.perf_counter() - start
                    self.counts["model_frames"] += len(group)
                    if offset == 0:
                        start = time.perf_counter()
                        direct_output = model(images[:1])
                        direct = select_successor_batch(direct_output[0], direct_output[4])[0]
                        batched = selected[0]
                        if (direct is None) != (batched is None):
                            raise RuntimeError("cache/direct availability mismatch")
                        if direct is not None:
                            if direct["anchor_index"] != batched["anchor_index"]:
                                raise RuntimeError("cache/direct anchor mismatch")
                            for field in ("box_xywh_input", "beliefs", "uncertainty", "objectness", "sigma2_norm_corrected"):
                                if not torch.allclose(direct[field], batched[field], rtol=1e-4, atol=1e-5):
                                    raise RuntimeError(f"cache/direct mismatch: {group[0][0]['key']}, {field}")
                        if sensor not in self.repeat_checked:
                            repeated = model(images[:1])
                            if not torch.equal(direct_output[0], repeated[0]):
                                raise RuntimeError("repeated fixed-input inference is not identical")
                            self.repeat_checked.add(sensor)
                            self.counts["repeat_checks"] += 1
                        torch.cuda.synchronize()
                        self.counts["verification_seconds"] += time.perf_counter() - start
                        self.counts["direct_checks"] += 1
                    for image_index, (item, candidate) in enumerate(zip(group, selected)):
                        frame, _pixels, geometry, clipping, seed = item
                        value = None if candidate is None else _candidate_record(candidate, images[image_index], geometry)[0]
                        output_rows[frame["key"]] = cache.sensor_record(value,
                            source_condition=condition.identifier, clipped_fraction=clipping, noise_seed_hex=seed)
        return output_rows


def conflict_diagnostics(rgb, thermal):
    b_rgb = np.array([0, 0] if rgb is None else rgb["beliefs"], dtype=np.float32)
    b_thermal = np.array([0, 0] if thermal is None else thermal["beliefs"], dtype=np.float32)
    u_rgb = 1.0 if rgb is None else rgb["semantic_uncertainty"]
    u_thermal = 1.0 if thermal is None else thermal["semantic_uncertainty"]
    conflict = float(np.clip(b_rgb.sum() * b_thermal.sum() - (b_rgb * b_thermal).sum(), 0, 1))
    p_rgb = b_rgb.astype(np.float64) + u_rgb / 2
    p_thermal = b_thermal.astype(np.float64) + u_thermal / 2
    probability_difference = np.abs(p_rgb - p_thermal).sum()
    degree = float(0.5 * probability_difference * (1 - u_rgb) * (1 - u_thermal))
    return conflict, degree


def diagnostics(root):
    counters = defaultdict(Counter)
    for row in cache.iter_cache(root):
        counts = counters[row["condition"]]
        counts["frames"] += 1
        counts["paired_present"] += int(row["paired_presence"])
        conflict, degree = conflict_diagnostics(row["rgb"]["candidate"], row["thermal"]["candidate"])
        counts["nonzero_conflict_frames"] += int(conflict > 0)
        counts["max_conflict_C"] = max(counts["max_conflict_C"], conflict)
        counts["max_degree_of_conflict"] = max(counts["max_degree_of_conflict"], degree)
        for sensor in cache.SENSORS:
            counts[f"{sensor}_missing"] += int(row[sensor]["candidate"] is None)
            counts[f"{sensor}_omitted"] += int(row[sensor]["omitted"])
            counts[f"{sensor}_clipped_fraction_sum"] += row[sensor]["clipped_fraction"]
    return {condition: dict(counts) for condition, counts in sorted(counters.items())}


def main():
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("import-validation", "clean-train", "pilot"), required=True)
    parser.add_argument("--phase2-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clean-validation-cache", type=Path)
    parser.add_argument("--frames-per-sequence", type=int, default=3)
    parser.add_argument("--sequence-limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.frames_per_sequence < 1 or (args.sequence_limit is not None and args.sequence_limit < 1):
        parser.error("workers, frame count and sequence limit must be positive")
    if args.mode != "pilot" and args.sequence_limit is not None:
        parser.error("sequence limits are restricted to the sampled pilot")
    repository = Path(__file__).resolve().parents[2]
    if _git(repository, "status", "--porcelain"):
        raise RuntimeError("cache provenance requires a clean repository")
    phase2, pair, phase5_path = contracts(args, repository)
    split = "train" if args.mode == "clean-train" else "val"
    grouped = select_frames(pair.development_manifest_path, split, pilot=args.mode == "pilot",
                            per_sequence=args.frames_per_sequence, sequence_limit=args.sequence_limit)
    selected_conditions = list(conditions()) if args.mode == "pilot" else [Condition()]
    descriptors = [descriptor(condition) for condition in selected_conditions]
    try:
        pytorch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pytorch_version = None  # CPU-only import/reading does not require Torch
    context = {
        "mode": args.mode, "split": split,
        "phase2_manifest_sha256": sha256_file(args.phase2_manifest),
        "corruption_specification_sha256": sha256_file(phase5_path),
        "development_manifest_sha256": pair.document["development_manifest"]["sha256"],
        "selected_keys_sha256": cache.digest([row["key"] for frames in grouped.values() for row in frames]),
        "conditions": descriptors,
        "source_sha256": {name: sha256_file(repository / name) for name in (
            "scripts/successor/build_detector_cache.py", "src/thesis_experiments/detector_cache.py")},
        "environment": {"numpy": np.__version__, "opencv": cv2.__version__,
                        "pytorch": pytorch_version},
    }
    parent_rows = {}
    if args.mode == "pilot":
        if args.clean_validation_cache is None:
            parser.error("pilot requires --clean-validation-cache")
        metadata = json.loads((args.clean_validation_cache / "CACHE.json").read_text())
        if (metadata["context"]["mode"] != "import-validation"
                or metadata["context"]["phase2_manifest_sha256"] != context["phase2_manifest_sha256"]
                or metadata["context"]["corruption_specification_sha256"] != context["corruption_specification_sha256"]):
            raise ValueError("pilot requires the frozen clean validation cache")
        context["clean_parent_context_id"] = metadata["context_id"]
        wanted = {row["key"] for frames in grouped.values() for row in frames}
        parent_rows = {row["key"]: row for row in cache.iter_cache(args.clean_validation_cache) if row["key"] in wanted}
        if set(parent_rows) != wanted:
            raise ValueError("clean parent omitted requested pilot frames")
    root = args.output_root.resolve()
    identifier = cache.initialize(root, context)
    imported = {}
    if args.mode == "import-validation":
        wanted = {row["key"] for frames in grouped.values() for row in frames}
        for sensor in cache.SENSORS:
            summary = json.loads(Path(phase2["extraction"][sensor]["summary"]["path"]).read_text())
            path = Path(summary["output"])
            if sha256_file(path) != summary["output_sha256"]:
                raise ValueError("canonical validation input hash mismatch")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            imported[sensor] = {row["key"]: row["candidate"] for row in rows}
            if len(rows) != len(imported[sensor]) or set(imported[sensor]) != wanted:
                raise ValueError("canonical validation input membership mismatch")
    engine = None
    run_counts = Counter()
    for condition in selected_conditions:
        desc = descriptor(condition)
        for sequence, frames in grouped.items():
            path = cache.shard_path(root, desc, sequence)
            if path.exists():
                cache.read_shard(path, desc, identifier, frames)
                run_counts["resumed_shards"] += 1
                continue
            sensors = {}
            for sensor in cache.SENSORS:
                if args.mode == "import-validation":
                    sensors[sensor] = {frame["key"]: cache.sensor_record(imported[sensor][frame["key"]]) for frame in frames}
                elif args.mode == "pilot" and condition.sensor != sensor:
                    sensors[sensor] = {frame["key"]: parent_rows[frame["key"]][sensor] for frame in frames}
                elif condition.corruption == "dropout":
                    sensors[sensor] = {frame["key"]: cache.sensor_record(None, omitted=True,
                                         source_condition=condition.identifier) for frame in frames}
                else:
                    if engine is None:
                        engine = Engine(pair, args.derived_root, args.workers)
                    sensors[sensor] = engine.infer(frames, sensor, condition)
            rows = [cache.make_row(frame, desc, {sensor: sensors[sensor][frame["key"]] for sensor in cache.SENSORS}, identifier)
                    for frame in frames]
            cache.write_shard(path, rows, desc, identifier, frames)
            run_counts["written_shards"] += 1
            if sum(run_counts.values()) % 10 == 0:
                print(f"condition={condition.identifier} sequence={sequence} "
                      f"written={run_counts['written_shards']} resumed={run_counts['resumed_shards']}", flush=True)
        print(f"condition={condition.identifier} written={run_counts['written_shards']} resumed={run_counts['resumed_shards']}", flush=True)
    contracts(args, repository)
    for relative, expected in context["source_sha256"].items():
        if sha256_file(repository / relative) != expected:
            raise RuntimeError("cache source changed while inference was running")
    index = cache.finalize(root, identifier, grouped, descriptors)
    support = diagnostics(root)
    seconds = time.perf_counter() - run_start
    byte_count = sum((root / shard["path"] / "rows.jsonl").stat().st_size for shard in index["shards"])
    report = {"rows": index["rows"], "bytes": byte_count, "run_wall_seconds": seconds,
              "git_commit": _git(repository, "rev-parse", "HEAD"),
              "inputs": {"phase2_manifest": str(args.phase2_manifest.resolve()),
                         "derived_root": str(args.derived_root.resolve())},
              "run_counts": dict(run_counts), "inference": {} if engine is None else dict(engine.counts),
              "bytes_per_paired_condition_row": byte_count / index["rows"],
              "projection_storage_bytes_per_10000_pairs_all33": byte_count / index["rows"] * 10000 * 33,
              "condition_support": support,
              "timing_scope": "this invocation only; resumed shards are verified/reused, not timed as fresh inference"}
    if engine is not None and engine.counts["model_frames"] and not run_counts["resumed_shards"]:
        report["model_seconds_per_image"] = engine.counts["model_seconds"] / engine.counts["model_frames"]
        report["projection_model_seconds_per_10000_pairs_all33"] = report["model_seconds_per_image"] * 10000 * 32
        report["projection_scope"] = "model forward time only, using equal-weight pilot conditions; excludes preparation/IO/verification/startup"
    if engine is not None:
        report["hardware"] = engine.hardware
    # Unique job report avoids overwriting the benchmark from a previous/resumed job.
    report_path = root / f"run_{time.time_ns()}.json"
    cache.immutable_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"cache_context_id={identifier}")
    print("DETECTOR CACHE BUILD PASSED")


if __name__ == "__main__":
    main()
