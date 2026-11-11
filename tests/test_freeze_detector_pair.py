import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path("scripts/successor/freeze_detector_pair.py").resolve()
SPEC = importlib.util.spec_from_file_location("freeze_detector_pair", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_checked_file_accepts_matching_hash(tmp_path):
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint")
    record = MODULE.checked_file(artifact, MODULE.sha256(artifact))
    assert record["size_bytes"] == 10
    assert len(record["sha256"]) == 64


def test_checked_file_rejects_hash_mismatch(tmp_path):
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        MODULE.checked_file(artifact, "0" * 64)


def test_atomic_create_refuses_overwrite(tmp_path):
    artifact = tmp_path / "manifest.json"
    MODULE.atomic_create(artifact, b"first")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        MODULE.atomic_create(artifact, b"second")
    assert artifact.read_bytes() == b"first"
