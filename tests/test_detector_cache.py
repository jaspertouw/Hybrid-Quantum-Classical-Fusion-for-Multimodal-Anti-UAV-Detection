import copy
import json

import pytest

from src.thesis_experiments import detector_cache as cache
from scripts.successor.build_detector_cache import select_frames, conflict_diagnostics


CLEAN = {"id": "clean", "corruption": "clean", "sensor": None, "severity": None}


def frame(number=1, sequence="sequence"):
    return {"key": f"val/{sequence}/{number:06d}", "split": "val", "sequence": sequence,
            "frame": number, "paired_presence": True, "attributes": ["SV"],
            "rgb": {"present": True, "box_xywh": [0, 0, 8, 4], "box_valid": True},
            "thermal": {"present": False, "box_xywh": None, "box_valid": False}}


def candidate():
    return {"beliefs": [0.6, 0.1], "semantic_uncertainty": 0.3,
            "objectness": 0.8, "candidate_score": 0.48, "anchor_index": 1,
            "box_xywh_input": [4, 4, 8, 4], "box_xywh_native": [0, 0, 8, 4],
            "sigma2_grid": [1, 1, 1, 1], "sigma2_pixel": [64, 64, 64, 64],
            "sigma2_norm_corrected_components": [1, 4, 1, 4],
            "sigma2_norm_inherited_head_components": [0.125, 0.25, 0.015625, 0.0625],
            "spatial_uncertainty_corrected": 2.5}


def create(root, frames=None):
    frames = [frame()] if frames is None else frames
    context = {"conditions": [CLEAN], "selected_keys_sha256": cache.digest(sorted(row["key"] for row in frames))}
    identifier = cache.initialize(root, context)
    rows = [cache.make_row(row, CLEAN, {"rgb": cache.sensor_record(candidate()),
                                       "thermal": cache.sensor_record(None)}, identifier) for row in frames]
    path = cache.shard_path(root, CLEAN, "sequence")
    marker = cache.write_shard(path, rows, CLEAN, identifier, frames)
    return frames, rows, path, identifier, marker


def test_roundtrip_preserves_labels_uncapped_values_and_vacuous_misses(tmp_path):
    frames, rows, _path, identifier, _marker = create(tmp_path)
    cache.finalize(tmp_path, identifier, {"sequence": frames}, [CLEAN])
    actual = list(cache.iter_cache(tmp_path))
    assert actual == rows
    assert actual[0]["ground_truth"]["rgb"] == frames[0]["rgb"]
    assert actual[0]["rgb"]["features_uncapped"]["spatial_uncertainty"] == 2.5
    assert actual[0]["thermal"]["features_uncapped"] == {
        "b_uav": 0.0, "semantic_uncertainty": 1.0, "objectness": 0.0, "spatial_uncertainty": None}


def test_resume_verifies_completed_shard_and_refuses_context_drift(tmp_path):
    frames, rows, path, identifier, marker = create(tmp_path)
    assert cache.read_shard(path, CLEAN, identifier, frames)[0] == rows
    assert cache.write_shard(path, rows, CLEAN, identifier, frames) == marker
    with pytest.raises(RuntimeError, match="metadata differs"):
        cache.initialize(tmp_path, {"different": "source"})


def test_identical_fresh_runs_have_identical_row_bytes(tmp_path):
    first = create(tmp_path / "first")[-1]
    second = create(tmp_path / "second")[-1]
    assert first["sha256"] == second["sha256"]


def test_partial_directory_and_missing_global_marker_are_not_accepted(tmp_path):
    frames, _rows, path, identifier, _marker = create(tmp_path)
    partial = tmp_path / "clean/.other.partial-interrupted"
    partial.mkdir()
    (partial / "rows.jsonl").write_text("partial bytes")
    with pytest.raises(FileNotFoundError):
        list(cache.iter_cache(tmp_path))
    cache.finalize(tmp_path, identifier, {"sequence": frames}, [CLEAN])
    assert len(list(cache.iter_cache(tmp_path))) == 1
    assert partial.is_dir()  # unrelated/interrupted data is not deleted automatically


def test_corrupt_later_shard_fails_before_first_row_is_yielded(tmp_path):
    frames, _rows, _path, identifier, _marker = create(tmp_path)
    other = frame(sequence="second")
    rows = [cache.make_row(other, CLEAN, {sensor: cache.sensor_record(None) for sensor in cache.SENSORS}, identifier)]
    path = cache.shard_path(tmp_path, CLEAN, "second")
    cache.write_shard(path, rows, CLEAN, identifier, [other])
    cache.finalize(tmp_path, identifier, {"sequence": frames, "second": [other]}, [CLEAN])
    (path / "rows.jsonl").write_text("broken\n")
    with pytest.raises(RuntimeError, match="corrupt"):
        next(cache.iter_cache(tmp_path))


def test_cache_rejects_modified_labels_and_incomplete_membership(tmp_path):
    frames, rows, _path, identifier, _marker = create(tmp_path)
    changed = copy.deepcopy(rows[0])
    changed["ground_truth"]["rgb"]["box_xywh"] = [1, 1, 8, 4]
    with pytest.raises(ValueError, match="changed original labels"):
        cache.validate_row(changed, CLEAN, identifier, frame=frames[0])
    with pytest.raises(ValueError, match="incomplete"):
        cache.write_shard(tmp_path / "new", [], CLEAN, identifier, frames)


def test_dropout_omits_only_target_sensor_without_spatial_imputation(tmp_path):
    condition = {"id": "rgb_dropout", "corruption": "dropout", "sensor": "rgb", "severity": None}
    row = cache.make_row(frame(), condition, {
        "rgb": cache.sensor_record(None, omitted=True, source_condition="rgb_dropout"),
        "thermal": cache.sensor_record(candidate()),
    }, "context")
    assert row["rgb"]["omitted"] is True
    assert row["rgb"]["features_uncapped"]["spatial_uncertainty"] is None
    assert row["thermal"]["candidate"] == candidate()


def test_pilot_sampling_uses_first_middle_last_without_predictions(tmp_path):
    path = tmp_path / "frames.jsonl"
    rows = [frame(i) for i in range(1, 11)]
    path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    selected = select_frames(path, "val", pilot=True, per_sequence=3)
    assert [row["frame"] for row in selected["sequence"]] == [1, 6, 10]


def test_conflict_distinguishes_identical_and_contradictory_opinions():
    rgb = {"beliefs": [0.9, 0], "semantic_uncertainty": 0.1}
    thermal = {"beliefs": [0, 0.9], "semantic_uncertainty": 0.1}
    conflict, degree = conflict_diagnostics(rgb, thermal)
    assert conflict == pytest.approx(0.81)
    assert degree == pytest.approx(0.729)
    assert conflict_diagnostics(rgb, None) == (0.0, 0.0)
    soft = {"beliefs": [0.6, 0.1], "semantic_uncertainty": 0.3}
    conflict, degree = conflict_diagnostics(soft, soft)
    assert conflict == pytest.approx(0.12)
    assert degree == 0


def test_cpu_import_pilot_reuse_and_resume_end_to_end(tmp_path, monkeypatch):
    from collections import Counter
    from types import SimpleNamespace
    from scripts.successor import build_detector_cache as builder

    frames = [frame(index) for index in range(1, 6)]
    manifest = tmp_path / "development.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in frames))
    phase2_manifest = tmp_path / "phase2.json"
    phase2_manifest.write_text("{}")
    spec = tmp_path / "corruptions.json"
    spec.write_text("{}")
    evidence = {"extraction": {}}
    for sensor in cache.SENSORS:
        source = tmp_path / f"{sensor}.jsonl"
        source.write_text("".join(json.dumps({"key": row["key"], "candidate": candidate()}) + "\n" for row in frames))
        summary = tmp_path / f"{sensor}_summary.json"
        summary.write_text(json.dumps({"output": str(source), "output_sha256": builder.sha256_file(source)}))
        evidence["extraction"][sensor] = {"summary": {"path": str(summary)}}
    pair = SimpleNamespace(development_manifest_path=manifest,
                           document={"development_manifest": {"sha256": builder.sha256_file(manifest)}})
    monkeypatch.setattr(builder, "contracts", lambda args, repository: (evidence, pair, spec))
    monkeypatch.setattr(builder, "_git", lambda *args: "")
    import sys

    def run(mode, output, extra=()):
        monkeypatch.setattr(sys, "argv", ["cache", "--mode", mode,
            "--phase2-manifest", str(phase2_manifest), "--dataset-root", str(tmp_path),
            "--derived-root", str(tmp_path), "--output-root", str(output), *extra])
        builder.main()

    clean_root = tmp_path / "clean"
    run("import-validation", clean_root)
    assert len(list(cache.iter_cache(clean_root))) == 5
    calls = []

    class FakeEngine:
        def __init__(self, *args):
            self.counts = Counter()  # no model timing, so no GPU-specific projection
            self.hardware = {"gpu": "synthetic"}

        def infer(self, rows, sensor, condition):
            calls.append((sensor, condition.identifier, len(rows)))
            return {row["key"]: cache.sensor_record(candidate(), source_condition=condition.identifier,
                    noise_seed_hex="0" * 32 if condition.corruption == "gaussian_noise" else None)
                    for row in rows}

    monkeypatch.setattr(builder, "Engine", FakeEngine)
    pilot_root = tmp_path / "pilot"
    extra = ("--clean-validation-cache", str(clean_root))
    run("pilot", pilot_root, extra)
    rows = list(cache.iter_cache(pilot_root))
    assert len(rows) == 3 * 33
    assert len(calls) == 30 and sum(count for _, _, count in calls) == 90
    assert all("dropout" not in condition and condition != "clean" for _, condition, _ in calls)
    original_hash = builder.sha256_file(pilot_root / "COMPLETED.json")
    run("pilot", pilot_root, extra)
    assert len(calls) == 30  # resume performs zero additional detector calls
    assert builder.sha256_file(pilot_root / "COMPLETED.json") == original_hash


def test_test_lock_fails_before_any_model_or_manifest_loading(tmp_path):
    from types import SimpleNamespace
    from scripts.successor.build_detector_cache import contracts

    (tmp_path / "test").mkdir()
    with pytest.raises(RuntimeError, match="test split"):
        contracts(SimpleNamespace(dataset_root=tmp_path), tmp_path)
