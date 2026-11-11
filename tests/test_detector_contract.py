import hashlib
import json
from pathlib import Path

import pytest

from src.thesis_experiments import detector_contract


def _write(path: Path, content: bytes) -> dict:
    path.write_bytes(content)
    return {
        "path": str(path),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _manifest(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    protocol = _write(tmp_path / "protocol.json", b"protocol")
    development = _write(tmp_path / "development.jsonl", b"development\n")
    rgb = _write(tmp_path / "rgb.pt", b"rgb")
    thermal = _write(tmp_path / "thermal.pt", b"thermal")
    rgb_clean = _write(tmp_path / "rgb-clean.pt", b"rgb-clean")
    thermal_clean = _write(tmp_path / "thermal-clean.pt", b"thermal-clean")
    monkeypatch.setattr(detector_contract, "EXPECTED_PROTOCOL_SHA256", protocol["sha256"])
    monkeypatch.setattr(detector_contract, "EXPECTED_DEVELOPMENT_SHA256", development["sha256"])
    document = {
        "schema_version": 1,
        "pair_id": detector_contract.EXPECTED_PAIR_ID,
        "protocol": protocol,
        "development_manifest": development,
        "modalities": {
            "rgb": {"selected": rgb, "clean": rgb_clean},
            "thermal": {"selected": thermal, "clean": thermal_clean},
        },
    }
    path = tmp_path / "pair.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def test_load_detector_pair_verifies_all_consumed_files(tmp_path, monkeypatch):
    path, _ = _manifest(tmp_path, monkeypatch)
    pair = detector_contract.load_detector_pair(
        path, expected_manifest_sha256=detector_contract.sha256_file(path)
    )
    assert set(pair.selected_checkpoints) == {"rgb", "thermal"}
    assert set(pair.clean_checkpoints) == {"rgb", "thermal"}
    assert pair.protocol_path.name == "protocol.json"


def test_load_detector_pair_rejects_modified_checkpoint(tmp_path, monkeypatch):
    path, document = _manifest(tmp_path, monkeypatch)
    expected_manifest = detector_contract.sha256_file(path)
    Path(document["modalities"]["rgb"]["selected"]["path"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="size mismatch"):
        detector_contract.load_detector_pair(
            path, expected_manifest_sha256=expected_manifest
        )


def test_load_detector_pair_rejects_wrong_manifest_hash(tmp_path, monkeypatch):
    path, _ = _manifest(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="manifest SHA-256 mismatch"):
        detector_contract.load_detector_pair(path, expected_manifest_sha256="0" * 64)
