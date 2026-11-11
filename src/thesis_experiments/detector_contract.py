"""Validation of the frozen detector pair used by successor experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .provenance import sha256_file


EXPECTED_PAIR_ID = "successor-detector-pair-v1"
EXPECTED_MANIFEST_SHA256 = (
    "4add7276627bd9700df04f3d844b7138040f31cac6578a71cb685db07beb63fc"
)
EXPECTED_PROTOCOL_SHA256 = (
    "a6eed103f1badf466f91209bf64134d370f5a077b5e1536f18bdb5620be62741"
)
EXPECTED_DEVELOPMENT_SHA256 = (
    "63d9d419e207b0ca46eb8821660061bdb66ef51cbe7420abba83328e6a3d371f"
)


@dataclass(frozen=True)
class DetectorPair:
    manifest_path: Path
    manifest_sha256: str
    protocol_path: Path
    development_manifest_path: Path
    selected_checkpoints: Mapping[str, Path]
    clean_checkpoints: Mapping[str, Path]
    document: Mapping[str, object]


def _verified_record(record: object, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"missing file record: {label}")
    try:
        path = Path(record["path"]).expanduser().resolve()
        expected_size = int(record["size_bytes"])
        expected_hash = str(record["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid file record: {label}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {label}: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch for {label}: {path}")
    return path


def load_detector_pair(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> DetectorPair:
    """Load the immutable pair manifest and verify every consumed file."""
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_sha256:
        raise RuntimeError(
            f"detector-pair manifest SHA-256 mismatch: {actual_manifest_hash}"
        )
    with manifest_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != 1:
        raise ValueError("unsupported detector-pair manifest schema")
    if document.get("pair_id") != EXPECTED_PAIR_ID:
        raise ValueError("unexpected detector-pair identity")

    protocol = document.get("protocol")
    development = document.get("development_manifest")
    if not isinstance(protocol, dict) or protocol.get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol identity mismatch")
    if not isinstance(development, dict) or development.get("sha256") != EXPECTED_DEVELOPMENT_SHA256:
        raise RuntimeError("development-manifest identity mismatch")

    modalities = document.get("modalities")
    if not isinstance(modalities, dict) or set(modalities) != {"rgb", "thermal"}:
        raise ValueError("manifest must contain exactly rgb and thermal modalities")
    selected = {
        modality: _verified_record(modalities[modality].get("selected"), f"{modality} selected checkpoint")
        for modality in ("rgb", "thermal")
    }
    clean = {
        modality: _verified_record(modalities[modality].get("clean"), f"{modality} clean checkpoint")
        for modality in ("rgb", "thermal")
    }
    return DetectorPair(
        manifest_path=manifest_path,
        manifest_sha256=actual_manifest_hash,
        protocol_path=_verified_record(protocol, "protocol"),
        development_manifest_path=_verified_record(development, "development manifest"),
        selected_checkpoints=selected,
        clean_checkpoints=clean,
        document=document,
    )


def load_phase2_manifest(manifest_path: Path, *, repository: Path) -> dict:
    """Verify the completed inference contract before a later cache/fusion run.

    Future consumers must use this gate as well as the immutable pair loader.
    Training logs remain archived evidence; routine consumers verify the scientific
    inputs, source files and canonical audit rather than rehashing every large log.
    """
    manifest_path = manifest_path.resolve(strict=True)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("status") != "phase2_complete":
        raise ValueError("Phase 2 is not complete")
    for key in ("detector_pair", "development_manifest", "inference_amendment", "canonical_audit"):
        _verified_record(document.get(key), key)
    pair = load_detector_pair(Path(document["detector_pair"]["path"]))
    if document["detector_pair"]["sha256"] != pair.manifest_sha256:
        raise RuntimeError("Phase 2 detector identity mismatch")
    repository = repository.resolve()
    for modality in ("rgb", "thermal"):
        summary_path = _verified_record(document["extraction"][modality]["summary"], f"{modality} summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for relative, digest in summary["source_provenance"]["source_sha256"].items():
            source = (repository / relative).resolve()
            if not source.is_relative_to(repository) or sha256_file(source) != digest:
                raise RuntimeError(f"frozen inference source changed: {relative}")
    return document
