#!/usr/bin/env python3
"""Validate Phase 2 evidence and create its immutable completion record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.thesis_experiments.detector_contract import load_detector_pair
from src.thesis_experiments.provenance import sha256_file


def record(path):
    path = path.resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing/empty evidence: {path}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def git(repository, *arguments):
    return subprocess.run(["git", *arguments], cwd=repository, check=True,
                          text=True, capture_output=True).stdout.strip()


def training_evidence(project, repository):
    jobs = {"clean_yolo": "28771695", "gaussian_heads": "29011888", "evidential_heads": "29055813"}
    scripts = {"clean_yolo": "train_clean_yolo.job", "gaussian_heads": "train_gaussian_heads.job",
               "evidential_heads": "train_evidential_heads.job"}
    evidence = {}
    for stage, job in jobs.items():
        logs = []
        commits = set()
        for task in (0, 1):
            path = project / "logs" / f"train_{stage}_{job}_{task}.out"
            with path.open(encoding="utf-8", errors="replace") as handle:
                header = handle.read(128 * 1024)
            match = re.search(r"Repository:.*?\(([0-9a-f]{40})\)", header)
            if match is None:
                raise ValueError(f"training commit missing from log: {path}")
            commits.add(match.group(1))
            logs.append(record(path))
        if len(commits) != 1:
            raise ValueError(f"modality training commits differ: {stage}")
        commit = commits.pop()
        historical_job = git(repository, "show", f"{commit}:scripts/successor/{scripts[stage]}")
        evidence[stage] = {
            "job": job, "commit": commit, "logs": logs,
            "released_training_tree": git(repository, "rev-parse", f"{commit}:resources/EDTC"),
            "submitted_job_source": historical_job,
        }
    history = []
    for modality in ("thermal", "rgb"):
        history.append(record(project / f"checkpoints/detectors/clean_yolo/{modality}_seed0/results.csv"))
        for stage in ("stage1g", "stage2"):
            history.append(record(project / f"checkpoints/detectors/evidential/{modality}_{stage}_seed0_history.csv"))
    return {"detector_seed": 0, "stages": evidence, "histories": history,
            "scope": "replacement pair; historical predecessor checkpoint identity is not claimed"}


def dataset_evidence(project, protocol):
    dataset = project / "data/Anti-UAV-RGBT"
    if (dataset / "test").exists():
        raise RuntimeError("test-set lock failed")
    preparation = project / "data/derived/yolo_inherited_v1/preparation_manifest.json"
    prepared = json.loads(preparation.read_text())
    manifest_attributes = {}
    with (project / "outputs/manifests/development_v1.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["split"], row["sequence"])
            tags = sorted(row["attributes"])
            if manifest_attributes.setdefault(key, tags) != tags:
                raise ValueError(f"inconsistent manifest attributes: {key}")
    lookup = {(row["split"], row["sequence"]): row for row in prepared}
    expected_sequences = sum(protocol["dataset"]["expected_sequence_counts"][split]
                             for split in ("train", "val"))
    if len(lookup) != len(prepared) or len(lookup) != expected_sequences:
        raise ValueError("invalid preparation membership")
    result = {"preparation_manifest": record(preparation), "splits": {}}
    for split in ("train", "val"):
        names = sorted(p.name for p in (dataset / split).iterdir() if p.is_dir())
        digest = hashlib.sha256("".join(name + "\n" for name in names).encode()).hexdigest()
        if digest != protocol["dataset"]["membership_sha256"][split]:
            raise ValueError(f"source sequence membership mismatch: {split}")
        attributes = dataset / f"label_new/{split}.json"
        attribute_mapping = json.loads(attributes.read_text())
        if set(attribute_mapping) != set(names):
            raise ValueError(f"attribute sequence membership mismatch: {split}")
        annotations = []
        for name in names:
            if sorted(attribute_mapping[name]) != manifest_attributes[(split, name)]:
                raise ValueError(f"source attributes differ from frozen manifest: {split}/{name}")
            entry = lookup[(split, name)]
            for modality, filename in (("thermal", "infrared.json"), ("rgb", "visible.json")):
                item = record(dataset / split / name / filename)
                if item["sha256"] != entry["modalities"][modality]["annotation_sha256"]:
                    raise ValueError(f"source annotation changed: {item['path']}")
                annotations.append(item)
        result["splits"][split] = {"sequences": len(names), "membership_sha256": digest,
                                   "attributes": record(attributes), "annotations": annotations}
    result["video_provenance"] = "original video hashes retained in preparation manifest; videos not re-decoded for freeze"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, repository = args.project_root.resolve(), args.repository.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if git(repository, "status", "--porcelain"):
        raise RuntimeError("refusing to freeze a dirty repository")
    pair = load_detector_pair(project / "outputs/manifests/detector_pair_v1.json")
    protocol = json.loads(pair.protocol_path.read_text())
    amendment = repository / "experiments/protocol/inference_v1.json"
    if json.loads(amendment.read_text())["amendment"]["status"] != "frozen":
        raise RuntimeError("inference amendment must be explicitly frozen first")
    canonical = project / "outputs/validation_candidates/v2"
    audit_path = canonical / "paired_validation_audit_v2.json"
    audit = json.loads(audit_path.read_text())
    for field, source in (("evaluator_sha256", "src/thesis_experiments/paired_evaluation.py"),
                          ("validator_sha256", "src/thesis_experiments/candidate_validation.py")):
        if audit.get("audit_source", {}).get(field) != sha256_file(repository / source):
            raise ValueError(f"audit does not match current scientific code: {source}")
    if audit["counts"]["frames"] != 61999 or len(audit["sequences"]) != 67:
        raise ValueError("incomplete canonical audit")
    if not audit["source_provenance_available"]:
        raise ValueError("canonical extraction source provenance is required")
    if audit["inference_amendment_sha256"] != sha256_file(amendment):
        raise ValueError("canonical audit uses a different amendment")
    extraction = {}
    parity = {}
    for modality in ("thermal", "rgb"):
        summary_path = canonical / f"{modality}_summary.json"
        summary = json.loads(summary_path.read_text())
        if summary["output_sha256"] != sha256_file(canonical / f"{modality}.jsonl"):
            raise ValueError("canonical output hash mismatch")
        extraction[modality] = {"summary": record(summary_path), "counts": summary["counts"]}
        parity_path = project / f"outputs/validation_candidates/v1/{modality}_parity_strict-fp32_batch128.json"
        report = json.loads(parity_path.read_text())
        if (not report["passed"] or report["counts"]["sampled_frames"] != 201
                or report.get("batch_size") != 128):
            raise ValueError(f"strict inference parity did not pass: {modality}")
        parity[modality] = record(parity_path)
    base = project / "checkpoints/base/yolov5s.pt"
    if sha256_file(base) != "8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee":
        raise ValueError("COCO initialization identity mismatch")
    payload = {
        "schema_version": 1, "status": "phase2_complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git(repository, "rev-parse", "HEAD"),
        "detector_pair": record(pair.manifest_path), "base_initialization": record(base),
        "inference_amendment": record(amendment), "canonical_audit": record(audit_path),
        "development_manifest": record(pair.development_manifest_path),
        "dataset": dataset_evidence(project, protocol),
        "training": training_evidence(project, repository),
        "extraction": extraction, "parity": parity,
        "environments": [record(project / f"outputs/environment/{name}-28770016.txt")
                         for name in ("detector-conda-explicit", "detector-pip-freeze")],
        "scope": "frozen replacement detectors and inference; feature fitting, fusion training, corruption pilots and final metrics remain later phases",
        "limitations": ["single detector seed", "evidence head supervises matched positive anchors",
                        "prior validation used in detector development", "test unavailable during Phase 2"],
    }
    content = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".phase2-freeze-", dir=args.output.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, args.output)  # atomic creation; fails if another writer won
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(f"phase2_manifest={args.output}")
    print(f"phase2_manifest_sha256={hashlib.sha256(content).hexdigest()}")
    print("PHASE 2 FREEZE PASSED")


if __name__ == "__main__":
    main()
