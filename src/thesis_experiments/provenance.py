"""Reproducibility metadata for successor experiment runs."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .config import ExperimentConfig


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git(repo: Path, *arguments: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_provenance(config: ExperimentConfig, inputs: Iterable[Path] = ()) -> dict:
    input_records = []
    for path in inputs:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"provenance input is not a file: {resolved}")
        input_records.append(
            {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}
        )

    status = _git(config.repository_root, "status", "--porcelain")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git(config.repository_root, "rev-parse", "HEAD"),
            "dirty": bool(status) if status is not None else None,
        },
        "protocol": {
            "path": str(config.protocol_path),
            "version": config.protocol_version,
            "sha256": sha256_file(config.protocol_path),
        },
        "runtime_config": config.portable_dict(),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "process_id": os.getpid(),
        },
        "inputs": input_records,
    }
