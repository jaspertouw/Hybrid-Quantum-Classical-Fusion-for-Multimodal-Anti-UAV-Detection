"""Versioned run layout and unambiguous completion state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import ExperimentConfig, REQUIRED_ARTIFACT_DIRS
from .io import atomic_write_json
from .provenance import collect_provenance, sha256_file


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class RunLayout:
    config: ExperimentConfig
    run_id: str

    def __post_init__(self) -> None:
        if not RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id must be 1-128 safe filename characters")

    @property
    def root(self) -> Path:
        return self.config.protocol_output_root / "runs" / self.run_id

    @property
    def provenance_path(self) -> Path:
        return self.root / "provenance.json"

    @property
    def completion_path(self) -> Path:
        return self.root / "COMPLETED.json"

    @property
    def partial_path(self) -> Path:
        return self.root / "PARTIAL.json"

    def artifact_dir(self, kind: str) -> Path:
        if kind not in REQUIRED_ARTIFACT_DIRS:
            raise ValueError(f"unknown artifact kind: {kind}")
        return self.root / self.config.artifact_layout[kind]

    def initialize(self, inputs: Iterable[Path] = ()) -> None:
        if self.completion_path.exists():
            raise RuntimeError(f"completed run is immutable: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        for kind in REQUIRED_ARTIFACT_DIRS:
            self.artifact_dir(kind).mkdir(exist_ok=True)
        atomic_write_json(self.provenance_path, collect_provenance(self.config, inputs))
        atomic_write_json(
            self.partial_path,
            {"schema_version": 1, "run_id": self.run_id, "state": "partial"},
        )

    def mark_complete(self, required_outputs: Iterable[Path]) -> None:
        if self.completion_path.exists():
            raise RuntimeError(f"run already completed: {self.root}")
        if not self.provenance_path.is_file() or not self.partial_path.is_file():
            raise RuntimeError("run must be initialized before it can be completed")

        records = []
        for output in required_outputs:
            resolved = output.expanduser().resolve()
            try:
                resolved.relative_to(self.root.resolve())
            except ValueError as exc:
                raise ValueError(f"required output is outside run directory: {resolved}") from exc
            if not resolved.is_file() or resolved.stat().st_size == 0:
                raise ValueError(f"required output is missing or empty: {resolved}")
            records.append(
                {"path": str(resolved.relative_to(self.root)), "size_bytes": resolved.stat().st_size,
                 "sha256": sha256_file(resolved)}
            )
        if not records:
            raise ValueError("at least one required output is needed to complete a run")

        atomic_write_json(
            self.completion_path,
            {"schema_version": 1, "run_id": self.run_id, "state": "complete", "outputs": records},
        )
        self.partial_path.unlink()

    def state(self) -> str:
        if self.completion_path.is_file():
            return "complete"
        if self.partial_path.is_file():
            return "partial"
        return "not_initialized"
