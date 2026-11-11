#!/usr/bin/env python3
"""Small Phase 5 smoke check: three validation pairs × 33 conditions only."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from lib.models.yolo_evidential.model import EvidentialYOLO
from lib.models.yolo_evidential.postprocess import select_successor_batch
from lib.utils.belief_fusion import compute_conflict
from scripts.successor.audit_validation_candidates import _load_validation_frames
from scripts.successor.extract_validation_candidates import _candidate_record
from src.thesis_experiments.candidate_validation import validate_candidate
from src.thesis_experiments.corruptions import Condition, FrameKey, conditions
from src.thesis_experiments.degraded_inputs import prepare_conditioned_sensor
from src.thesis_experiments.detector_contract import load_phase2_manifest, load_detector_pair
from src.thesis_experiments.inference_runtime import configure_inference
from src.thesis_experiments.io import atomic_write_json
from src.thesis_experiments.provenance import sha256_file


def sample_frames(frames):
    grouped = defaultdict(list)
    for frame in frames:
        grouped[frame["sequence"]].append(frame)
    sequences = sorted(grouped)
    selected = []
    for index in (0, len(sequences) // 2, len(sequences) - 1):
        rows = sorted(grouped[sequences[index]], key=lambda row: row["frame"])
        selected.append(rows[len(rows) // 2])
    return selected


def image_panel(clean, rows):
    families = (("rgb", "gaussian_noise"), ("thermal", "gaussian_noise"),
                ("rgb", "gaussian_blur"), ("thermal", "gaussian_blur"),
                ("rgb", "illumination"), ("thermal", "contrast"))
    tile_w, tile_h = 256, 208
    panel = np.full((7 * tile_h, 6 * tile_w, 3), 245, dtype=np.uint8)

    def tile(image, label, row, column):
        x, y = column * tile_w, row * tile_h
        cv2.putText(panel, label, (x + 5, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (20, 20, 20), 1, cv2.LINE_AA)
        if image is None:
            cv2.putText(panel, "OMITTED: no model call", (x + 6, y + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 180), 1)
            return
        rgb = np.clip(image.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        scale = min((tile_w - 8) / rgb.shape[1], (tile_h - 38) / rgb.shape[0])
        width, height = round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)
        preview = cv2.resize(rgb[:, :, ::-1], (width, height), interpolation=cv2.INTER_AREA)
        panel[y + 32:y + 32 + height, x + 4:x + 4 + width] = preview

    for row, (sensor, corruption) in enumerate(families):
        tile(clean[sensor], f"{sensor}: clean", row, 0)
        for severity in range(1, 6):
            condition = Condition(corruption, sensor, severity)
            tile(rows[condition.identifier][sensor], f"{sensor}: {corruption} {severity}", row, severity)
    tile(None, "RGB dropout (display placeholder)", 6, 0)
    tile(clean["thermal"], "Thermal remains unchanged", 6, 1)
    tile(None, "Thermal dropout (display placeholder)", 6, 3)
    tile(clean["rgb"], "RGB remains unchanged", 6, 4)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-manifest", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to replace preview artifacts: {args.output_dir}")
    repository = Path(__file__).resolve().parents[2]
    spec_path = repository / "experiments/protocol/corruptions_v1.json"
    spec = json.loads(spec_path.read_text())
    if sha256_file(args.phase2_manifest) != spec["phase2_completion_sha256"]:
        raise ValueError("unexpected Phase 2 completion record")
    frozen = load_phase2_manifest(args.phase2_manifest, repository=repository)
    pair = load_detector_pair(Path(frozen["detector_pair"]["path"]))
    frames = sample_frames(_load_validation_frames(pair.development_manifest_path))
    configure_inference()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device("cuda:0")
    models = {}
    for sensor in ("rgb", "thermal"):
        models[sensor] = EvidentialYOLO.load(str(pair.selected_checkpoints[sensor]),
                                            str(pair.clean_checkpoints[sensor]), device=device, K=2).eval()
        models[sensor].model[-1].use_cls_for_conf = False
    args.output_dir.mkdir(parents=True)
    counts = Counter()
    results = []
    previews = []
    with torch.inference_mode():
        for frame in frames:
            key = FrameKey(frame["split"], frame["sequence"], frame["frame"])
            paths = {sensor: args.derived_root / frame[sensor]["image"] for sensor in models}
            clean = {sensor: prepare_conditioned_sensor(paths[sensor], sensor, key, Condition(),
                     stride=int(model.stride.max())).pixels_chw for sensor, model in models.items()}
            pixels = {}
            for condition in conditions():
                entry = {"key": frame["key"], "condition": condition.identifier,
                         "paired_presence": frame["paired_presence"], "sensors": {}}
                pixels[condition.identifier] = {}
                opinions = {}
                for sensor, model in models.items():
                    prepared = prepare_conditioned_sensor(paths[sensor], sensor, key, condition,
                                                          stride=int(model.stride.max()))
                    if prepared is None:
                        if condition != Condition("dropout", sensor):
                            raise RuntimeError("unexpected omitted sensor")
                        counts["dropout_model_calls_skipped"] += 1
                        entry["sensors"][sensor] = {"candidate": None, "omitted": True}
                        opinions[sensor] = (np.zeros(2), 1.0)
                        pixels[condition.identifier][sensor] = None
                        continue
                    image = prepared.pixels_chw
                    pixels[condition.identifier][sensor] = image
                    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                        raise RuntimeError("prepared pixels outside [0,1]")
                    if sensor != condition.sensor:
                        np.testing.assert_array_equal(image, clean[sensor])
                        counts["unchanged_sensor_checks"] += 1
                    if condition.corruption == "gaussian_noise" and sensor == condition.sensor:
                        repeat = prepare_conditioned_sensor(paths[sensor], sensor, key, condition,
                                                            stride=int(model.stride.max()))
                        np.testing.assert_array_equal(image, repeat.pixels_chw)
                        counts["noise_repeat_checks"] += 1
                    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
                    output = model(tensor)  # already [0,1]; never divide by 255 twice
                    selected = select_successor_batch(output[0], output[4])[0]
                    counts["detector_calls"] += 1
                    candidate = None if selected is None else _candidate_record(
                        selected, tensor[0], prepared.geometry)[0]
                    validate_candidate(candidate)
                    opinions[sensor] = ((np.zeros(2), 1.0) if candidate is None else
                                        (np.array(candidate["beliefs"]), candidate["semantic_uncertainty"]))
                    entry["sensors"][sensor] = {
                        "candidate": candidate, "omitted": False,
                        "clipped_fraction": prepared.clipped_fraction,
                        "noise_seed_hex": prepared.noise_seed_hex,
                    }
                    counts["selected" if candidate else "detector_missing"] += 1
                b_rgb, u_rgb = opinions["rgb"]
                b_thermal, u_thermal = opinions["thermal"]
                entry["conflict_C"] = float(compute_conflict(
                    torch.tensor(b_rgb, dtype=torch.float32),
                    torch.tensor(b_thermal, dtype=torch.float32)).item())
                entry["degree_of_conflict"] = float(0.5 * np.abs(
                    b_rgb + u_rgb / 2 - b_thermal - u_thermal / 2).sum() * (1 - u_rgb) * (1 - u_thermal))
                results.append(entry)
            path = args.output_dir / f"{frame['sequence']}_{frame['frame']:06d}.png"
            if not cv2.imwrite(str(path), image_panel(clean, pixels)):
                raise RuntimeError(f"could not write visual examples: {path}")
            previews.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
            print(f"verified: {frame['key']} across all 33 conditions", flush=True)
    if len(results) != 99 or counts["detector_calls"] != 192 or counts["dropout_model_calls_skipped"] != 6:
        raise RuntimeError("unexpected smoke-check workload")
    sources = ["src/thesis_experiments/corruptions.py", "src/thesis_experiments/degraded_inputs.py",
               "scripts/successor/verify_corruptions.py", "resources/EDTC/lib/utils/belief_fusion.py"]
    report = {
        "schema_version": 1, "passed": True, "scope": "3 validation pairs only; no performance/calibration claim",
        "phase2_manifest_sha256": sha256_file(args.phase2_manifest),
        "corruption_specification_sha256": sha256_file(spec_path),
        "source_sha256": {name: sha256_file(repository / name) for name in sources},
        "environment": {"numpy": np.__version__, "opencv": cv2.__version__, "pytorch": torch.__version__},
        "sample_rule": "middle frame of first/middle/last validation sequence, selected before inference",
        "counts": dict(counts), "previews": previews, "results": results,
    }
    atomic_write_json(args.output_dir / "verification.json", report)
    print(json.dumps({"pairs": len(results), "counts": dict(counts), "passed": True}, indent=2))
    print("PHASE 5 SMALL VALIDATION CHECK PASSED")


if __name__ == "__main__":
    main()
