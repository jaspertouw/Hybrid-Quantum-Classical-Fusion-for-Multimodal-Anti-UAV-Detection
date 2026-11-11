#!/usr/bin/env python3
"""Bounded development pilot for background-supervised frame opinions.

No production cache is written. Failure of scientific gates is reported explicitly;
successful script execution alone does not authorize the revised primary protocol.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from src.thesis_experiments import presence_opinions as opinions
from src.thesis_experiments.corruptions import Condition
from src.thesis_experiments.provenance import sha256_file, _git
from scripts.successor.build_detector_cache import contracts, select_frames


def run(args):
    import torch
    from src.thesis_experiments.inference_runtime import configure_inference
    from src.thesis_experiments.degraded_inputs import prepare_conditioned_sensor
    for root in (REPO / "resources/EDTC", REPO / "resources/EDTC/yolov5"):
        sys.path.insert(0, str(root))
    from lib.models.yolo_evidential.model import EvidentialYOLO
    from src.thesis_experiments.corruptions import FrameKey

    if _git(REPO, "status", "--porcelain"):
        raise RuntimeError("pilot requires a clean committed repository")
    phase2, pair, phase5_path = contracts(args, REPO)
    spec_path = REPO / "experiments/protocol/presence_opinions_v1.json"
    spec = json.loads(spec_path.read_text())
    if spec["status"] != "development" or spec["phase2_completion_sha256"] != sha256_file(args.phase2_manifest):
        raise ValueError("opinion pilot contract mismatch")
    args.output_root.mkdir(parents=True, exist_ok=False)
    configure_inference()
    torch.set_num_threads(4)
    all_frames = [json.loads(line) for line in pair.development_manifest_path.read_text().splitlines()]
    train = [row for row in all_frames if row["split"] == "train"]
    val = [row for row in all_frames if row["split"] == "val"]
    grouped = select_frames(pair.development_manifest_path, "val", pilot=True, per_sequence=3)
    paired = [row for rows in grouped.values() for row in rows]
    sources = {str(path.relative_to(REPO)): sha256_file(path) for path in
               (Path(__file__).resolve(), REPO / "src/thesis_experiments/presence_opinions.py", spec_path)}
    report = {"status": "pilot_only", "git_commit": _git(REPO, "rev-parse", "HEAD"),
              "phase2_sha256": sha256_file(args.phase2_manifest),
              "phase5_sha256": sha256_file(phase5_path), "source_sha256": sources,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "modalities": {}, "conditions": {}, "sampling": "SHA256 key-ranked within sensor-specific class"}
    predictions = {}
    for sensor in ("rgb", "thermal"):
        torch.manual_seed(spec["seed"])
        np.random.seed(spec["seed"])
        training = opinions.sample_by_class(train, sensor, spec["pilot_per_class"], split="train", seed=spec["seed"])
        validation = opinions.sample_by_class(val, sensor, spec["pilot_per_class"], split="val", seed=spec["seed"])
        detector = EvidentialYOLO.load(str(pair.selected_checkpoints[sensor]),
                    str(pair.clean_checkpoints[sensor]), device=torch.device("cuda:0"), K=2).eval()
        extractor = opinions.FrozenFrameFeatures(detector)
        stride = int(detector.stride.max())

        def extract(rows, condition=Condition()):
            buffers = []
            for offset in range(0, len(rows), 32):
                chunk = rows[offset:offset + 32]
                groups = {}
                for index, row in enumerate(chunk):
                    path = (args.derived_root / row[sensor]["image"]).resolve()
                    if not path.is_relative_to(args.derived_root.resolve()):
                        raise ValueError("image outside derived root")
                    prepared = prepare_conditioned_sensor(path, sensor,
                        FrameKey(row["split"], row["sequence"], row["frame"]), condition, stride=stride)
                    if prepared is None:
                        raise RuntimeError("dropout must bypass feature extraction")
                    groups.setdefault(prepared.pixels_chw.shape, []).append((index, prepared.pixels_chw))
                outputs = [None] * len(chunk)
                for group in groups.values():
                    images = torch.from_numpy(np.stack([item[1] for item in group])).cuda()
                    values = extractor(images).cpu().numpy()
                    for (index, _), value in zip(group, values):
                        outputs[index] = value
                buffers.extend(outputs)
            return np.stack(buffers)

        train_x, val_x = extract(training), extract(validation)
        mean, scale = opinions.fit_standardization(train_x, split="train")
        transform = lambda values: torch.from_numpy(((values - mean) / scale).astype(np.float32)).cuda()
        train_x, val_x = transform(train_x), transform(val_x)
        labels = torch.tensor([row[sensor]["present"] for row in training], device="cuda")
        prevalence = sum(row[sensor]["present"] for row in train) / len(train)
        val_prevalence = sum(row[sensor]["present"] for row in val) / len(val)
        head = opinions.build_head(train_x.shape[1], spec["hidden_size"]).cuda()
        optimizer = torch.optim.Adam(head.parameters(), lr=spec["learning_rate"])
        history = []
        # Fixed epochs; validation outcomes cannot select the checkpoint in this pilot.
        for epoch in range(1, spec["epochs"] + 1):
            head.train()
            optimizer.zero_grad()
            loss = opinions.evidential_loss(head(train_x), labels, epoch=epoch,
                                             population_prevalence=prevalence)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite frame-opinion loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10)
            optimizer.step()
            history.append(float(loss.detach()))
        head.eval()
        with torch.no_grad():
            evidence = head(val_x).cpu().numpy()
        metric = opinions.classification_report(evidence,
                  [row[sensor]["present"] for row in validation], population_prevalence=val_prevalence)
        metric.update({"train_prevalence": prevalence, "validation_prevalence": val_prevalence,
                       "history": history, "train_keys": [row["key"] for row in training],
                       "validation_keys": [row["key"] for row in validation]})
        # Representative paired validation, then the predeclared moderate corruptions.
        predictions[sensor] = {}
        for condition in (Condition(), Condition("gaussian_noise", sensor, 3),
                          Condition("gaussian_blur", sensor, 3)):
            with torch.no_grad():
                features = transform(extract(paired, condition))
                predicted = head(features)
                repeated = head(features)
                if not torch.equal(predicted, repeated):
                    raise RuntimeError("head repeat check failed")
                predictions[sensor][condition.identifier] = predicted.cpu().numpy()
        saved = {"state_dict": head.cpu().state_dict(), "mean": mean.tolist(), "scale": scale.tolist(),
                 "input_size": train_x.shape[1], "specification": spec, "sensor": sensor,
                 "phase2_sha256": report["phase2_sha256"], "source_sha256": sources,
                 "train_keys": metric["train_keys"], "status": "pilot_only"}
        checkpoint = args.output_root / f"{sensor}_pilot.pt"
        torch.save(saved, checkpoint)
        metric["checkpoint_sha256"] = sha256_file(checkpoint)
        report["modalities"][sensor] = metric
        extractor.close()
        del extractor, detector, head, optimizer, train_x, val_x
        torch.cuda.empty_cache()
        print(f"{sensor}: balanced_accuracy={metric['balanced_accuracy']:.4f} "
              f"both_class_learning_gate={metric['both_class_learning_gate']}", flush=True)
    for condition in [Condition()] + [Condition(family, sensor, 3)
                         for sensor in ("rgb", "thermal") for family in ("gaussian_noise", "gaussian_blur")]:
        rgb = predictions["rgb"][condition.identifier if condition.sensor == "rgb" else "clean"]
        thermal = predictions["thermal"][condition.identifier if condition.sensor == "thermal" else "clean"]
        report["conditions"][condition.identifier] = opinions.paired_report(rgb, thermal)
    report["paired_keys"] = [row["key"] for row in paired]
    report["both_class_learning_gate"] = all(item["both_class_learning_gate"] for item in report["modalities"].values())
    report["confident_opposition_observed"] = any(item["confident_opposite_decisions"] > 0
                                                 for item in report["conditions"].values())
    report["primary_ready"] = False  # Requires review, broader training and cache/schema amendment.
    contracts(args, REPO)
    for relative, expected in sources.items():
        if sha256_file(REPO / relative) != expected:
            raise RuntimeError("pilot source changed during execution")
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in
        ("both_class_learning_gate", "confident_opposition_observed", "primary_ready", "conditions")}, indent=2))
    print("FRAME OPINION PILOT EXECUTION PASSED; SCIENTIFIC GATES REQUIRE REVIEW")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args())
