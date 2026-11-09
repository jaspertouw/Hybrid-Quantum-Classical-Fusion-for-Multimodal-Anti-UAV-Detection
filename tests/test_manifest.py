import json
from pathlib import Path

import pytest

from src.thesis_experiments import manifest


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def miniature_dataset(root: Path) -> None:
    for split, sequence in (("train", "train_seq"), ("val", "val_seq")):
        sequence_dir = root / split / sequence
        annotation = {"exist": [1, 0], "gt_rect": [[1, 2, 3, 4], []]}
        write_json(sequence_dir / "infrared.json", annotation)
        write_json(sequence_dir / "visible.json", annotation)
        write_json(root / "label_new" / f"{split}.json", {sequence: ["SV"]})


def test_manifest_preserves_boxes_presence_attributes_and_hash(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    miniature_dataset(dataset)
    monkeypatch.setattr(manifest, "EXPECTED_SEQUENCE_COUNTS", {"train": 1, "val": 1})
    output, summary_path = tmp_path / "manifest.jsonl", tmp_path / "summary.json"

    summary = manifest.build_development_manifest(dataset, output, summary_path)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert [row["key"] for row in rows] == [
        "train/train_seq/000001", "train/train_seq/000002",
        "val/val_seq/000001", "val/val_seq/000002",
    ]
    assert rows[0]["paired_presence"] is True
    assert rows[0]["thermal"]["box_xywh"] == [1.0, 2.0, 3.0, 4.0]
    assert rows[0]["attributes"] == ["SV"]
    assert rows[1]["paired_presence"] is False
    assert rows[1]["thermal"]["box_xywh"] is None
    assert rows[1]["thermal"]["box_valid"] is False
    assert summary["splits"]["train"]["frames"] == 2
    assert summary["splits"]["train"]["paired_present"] == 1
    assert summary_path.is_file()
    assert len(summary["manifest_sha256"]) == 64


def test_manifest_rejects_test_split(tmp_path):
    with pytest.raises(ValueError, match="forbids"):
        manifest.build_development_manifest(
            tmp_path, tmp_path / "manifest.jsonl", tmp_path / "summary.json", splits=("test",)
        )


def test_manifest_rejects_sensor_length_mismatch(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    miniature_dataset(dataset)
    write_json(
        dataset / "train" / "train_seq" / "visible.json",
        {"exist": [1], "gt_rect": [[1, 2, 3, 4]]},
    )
    monkeypatch.setattr(manifest, "EXPECTED_SEQUENCE_COUNTS", {"train": 1, "val": 1})
    with pytest.raises(ValueError, match="sensor frame-count mismatch"):
        manifest.build_development_manifest(
            dataset, tmp_path / "manifest.jsonl", tmp_path / "summary.json"
        )
