"""Central configuration loading with portable path overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


REQUIRED_ARTIFACT_DIRS = ("manifests", "caches", "models", "metrics", "plots", "logs")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    repository_root: Path
    runtime_config_version: int
    protocol_path: Path
    dataset_root: Path
    checkpoint_root: Path
    output_root: Path
    artifact_layout: Mapping[str, str]

    @property
    def protocol(self) -> dict:
        with self.protocol_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    @property
    def protocol_version(self) -> str:
        return str(self.protocol["protocol"]["version"])

    @property
    def protocol_output_root(self) -> Path:
        return self.output_root / f"protocol-{self.protocol_version}"

    def portable_dict(self) -> dict:
        """Serializable resolved configuration recorded in run provenance."""
        return {
            "runtime_config_version": self.runtime_config_version,
            "source_path": str(self.source_path),
            "protocol_path": str(self.protocol_path),
            "protocol_version": self.protocol_version,
            "dataset_root": str(self.dataset_root),
            "checkpoint_root": str(self.checkpoint_root),
            "output_root": str(self.output_root),
            "artifact_layout": dict(self.artifact_layout),
        }


def load_experiment_config(
    config_path: Path = Path("experiments/config/default.json"),
    *,
    dataset_root: Optional[Path] = None,
    checkpoint_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    root: Optional[Path] = None,
) -> ExperimentConfig:
    repo = (root or repository_root()).resolve()
    source = config_path.expanduser()
    if not source.is_absolute():
        source = repo / source
    source = source.resolve()
    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    if raw.get("runtime_config_version") != 1:
        raise ValueError("unsupported runtime_config_version")
    layout = raw.get("artifact_layout", {})
    missing = set(REQUIRED_ARTIFACT_DIRS) - set(layout)
    if missing:
        raise ValueError(f"artifact_layout is missing: {sorted(missing)}")
    if len(set(layout.values())) != len(layout):
        raise ValueError("artifact directory names must be unique")
    if any(Path(value).is_absolute() or len(Path(value).parts) != 1 for value in layout.values()):
        raise ValueError("artifact directory names must be single relative path components")

    def selected(override: Optional[Path], key: str) -> Path:
        return override.expanduser().resolve() if override else _resolve(repo, raw[key])

    protocol_path = _resolve(repo, raw["protocol_path"])
    if not protocol_path.is_file():
        raise ValueError(f"protocol file does not exist: {protocol_path}")

    return ExperimentConfig(
        source_path=source,
        repository_root=repo,
        runtime_config_version=raw["runtime_config_version"],
        protocol_path=protocol_path,
        dataset_root=selected(dataset_root, "dataset_root"),
        checkpoint_root=selected(checkpoint_root, "checkpoint_root"),
        output_root=selected(output_root, "output_root"),
        artifact_layout=dict(layout),
    )
