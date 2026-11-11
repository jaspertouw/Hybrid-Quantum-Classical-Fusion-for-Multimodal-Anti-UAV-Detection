import hashlib
import json

import pytest

from scripts.successor.finalize_phase2 import dataset_evidence, record


def test_evidence_paths_resolve_symlinked_home(tmp_path):
    real = tmp_path / "gpfs"
    real.mkdir()
    artifact = real / "artifact.txt"
    artifact.write_text("evidence")
    alias = tmp_path / "home"
    alias.symlink_to(real, target_is_directory=True)
    assert record(alias / "artifact.txt") == record(artifact)


def test_dataset_freeze_checks_attributes_against_manifest(tmp_path):
    dataset = tmp_path / "data/Anti-UAV-RGBT"
    prepared = []
    rows = []
    protocol = {"dataset": {"expected_sequence_counts": {"train": 1, "val": 1},
                             "membership_sha256": {}}}
    for split in ("train", "val"):
        directory = dataset / split / "sequence"
        directory.mkdir(parents=True)
        modalities = {}
        for modality, filename in (("thermal", "infrared.json"), ("rgb", "visible.json")):
            path = directory / filename
            path.write_text('{"exist":[1],"gt_rect":[[0,0,2,2]]}')
            modalities[modality] = {"annotation_sha256": record(path)["sha256"]}
        prepared.append({"split": split, "sequence": "sequence", "modalities": modalities})
        rows.append({"split": split, "sequence": "sequence", "attributes": ["SV"]})
        attributes = dataset / f"label_new/{split}.json"
        attributes.parent.mkdir(exist_ok=True)
        attributes.write_text(json.dumps({"sequence": ["SV"]}))
        protocol["dataset"]["membership_sha256"][split] = hashlib.sha256(b"sequence\n").hexdigest()
    preparation = tmp_path / "data/derived/yolo_inherited_v1/preparation_manifest.json"
    preparation.parent.mkdir(parents=True)
    preparation.write_text(json.dumps(prepared))
    manifest = tmp_path / "outputs/manifests/development_v1.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = dataset_evidence(tmp_path, protocol)
    assert result["splits"]["train"]["sequences"] == 1
    (dataset / "label_new/val.json").write_text(json.dumps({"sequence": ["OV"]}))
    with pytest.raises(ValueError, match="source attributes differ"):
        dataset_evidence(tmp_path, protocol)
