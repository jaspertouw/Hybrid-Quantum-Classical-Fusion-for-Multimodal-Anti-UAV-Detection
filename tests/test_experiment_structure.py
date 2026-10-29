import json
from pathlib import Path

import pytest

from src.thesis_experiments.config import REQUIRED_ARTIFACT_DIRS, load_experiment_config
from src.thesis_experiments.io import atomic_write_json
from src.thesis_experiments.run import RunLayout


def test_config_paths_and_cli_overrides_are_portable(tmp_path):
    config = load_experiment_config(output_root=tmp_path / "custom-output")
    assert config.output_root == (tmp_path / "custom-output").resolve()
    assert config.protocol_version == "1.0.0"
    assert config.protocol_output_root.name == "protocol-1.0.0"


def test_run_layout_records_provenance_and_starts_partial(tmp_path):
    config = load_experiment_config(output_root=tmp_path)
    run = RunLayout(config, "unit-test-run")
    run.initialize([Path("experiments/protocol/v1.json")])

    assert run.state() == "partial"
    assert run.partial_path.is_file()
    assert not run.completion_path.exists()
    assert all(run.artifact_dir(kind).is_dir() for kind in REQUIRED_ARTIFACT_DIRS)
    provenance = json.loads(run.provenance_path.read_text(encoding="utf-8"))
    assert provenance["protocol"]["version"] == "1.0.0"
    assert len(provenance["protocol"]["sha256"]) == 64
    assert provenance["inputs"][0]["sha256"] == provenance["protocol"]["sha256"]


def test_completion_requires_nonempty_outputs_inside_run(tmp_path):
    config = load_experiment_config(output_root=tmp_path)
    run = RunLayout(config, "completion-test")
    run.initialize()

    missing = run.artifact_dir("metrics") / "missing.json"
    with pytest.raises(ValueError, match="missing or empty"):
        run.mark_complete([missing])
    outside = tmp_path / "outside.json"
    atomic_write_json(outside, {"value": 1})
    with pytest.raises(ValueError, match="outside run directory"):
        run.mark_complete([outside])

    result = run.artifact_dir("metrics") / "result.json"
    atomic_write_json(result, {"metric": 0.5})
    run.mark_complete([result])
    assert run.state() == "complete"
    assert not run.partial_path.exists()
    marker = json.loads(run.completion_path.read_text(encoding="utf-8"))
    assert marker["outputs"][0]["path"] == "metrics/result.json"
    assert len(marker["outputs"][0]["sha256"]) == 64


def test_completed_run_cannot_be_reinitialized(tmp_path):
    config = load_experiment_config(output_root=tmp_path)
    run = RunLayout(config, "immutable-complete-run")
    run.initialize()
    result = run.artifact_dir("metrics") / "result.json"
    atomic_write_json(result, {"ok": True})
    run.mark_complete([result])
    with pytest.raises(RuntimeError, match="immutable"):
        run.initialize()


@pytest.mark.parametrize("run_id", ["../escape", "has spaces", "", "/absolute"])
def test_unsafe_run_ids_are_rejected(run_id, tmp_path):
    config = load_experiment_config(output_root=tmp_path)
    with pytest.raises(ValueError):
        RunLayout(config, run_id)
