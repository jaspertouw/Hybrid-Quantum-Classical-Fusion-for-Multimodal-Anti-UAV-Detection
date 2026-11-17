"""Operational contracts for development jobs, without accessing test data."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
from src.thesis_experiments import jobs


def test_interruption_retry_reuse_and_corruption(tmp_path):
    root = tmp_path / "rehearsal"
    jobs.create_plan(root, "rehearsal")
    assert jobs.prepare(root) == "prepared"
    assert jobs.prepare(root) == "reused"
    with pytest.raises(RuntimeError, match="Injected"):
        jobs.run_case(root, 0, True)
    assert not (root / "models/mlp-00/COMPLETED.json").exists()
    with pytest.raises(FileNotFoundError):
        jobs.aggregate(root)
    assert jobs.run_case(root, 0) == "completed"
    marker, folder = jobs.completed_case(root, 0)
    original = (folder / "COMPLETED.json").read_bytes()
    assert jobs.run_case(root, 0) == "reused"
    assert (folder / "COMPLETED.json").read_bytes() == original
    assert jobs.run_case(root, 1) == "completed"
    assert jobs.aggregate(root) == "aggregated"
    assert jobs.aggregate(root) == "reused"
    # Compare a retried fit with a fresh fit using the same data and seed.
    control = tmp_path / "control"
    jobs.create_plan(control, "rehearsal")
    jobs.prepare(control)
    jobs.run_case(control, 0)
    fresh, fresh_folder = jobs.completed_case(control, 0)
    def predictions(base, record):
        return base / record["result"].replace("result.json", "predictions.npz")
    with np.load(predictions(folder, marker)) as a, np.load(predictions(fresh_folder, fresh)) as b:
        np.testing.assert_array_equal(a["probability"], b["probability"])
    model = folder / marker["result"].replace("result.json", "model.pt")
    model.write_bytes(model.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corruption"):
        jobs.run_case(root, 0)
    with pytest.raises(ValueError, match="corruption"):
        jobs.aggregate(root)


def test_plan_and_lock_guards(tmp_path):
    root = tmp_path / "run"
    plan = jobs.create_plan(root, "rehearsal")
    with pytest.raises(ValueError, match="index"):
        jobs.case_at(plan, -1)
    lock = root / ".test.lock"
    with jobs.task_lock(lock, plan["plan_id"]):
        with pytest.raises(RuntimeError, match="locked"):
            with jobs.task_lock(lock, plan["plan_id"]):
                pass
        with pytest.raises(RuntimeError):
            jobs.unlock_inactive(lock, plan["plan_id"])
    record = json.loads((root / "PLAN.json").read_text())
    record["plan"]["training_cap"] = 1
    (root / "PLAN.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="drift"):
        jobs.load_plan(root)
