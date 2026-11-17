"""Immutable development task plans, verified reuse and deterministic restart.

Incomplete fits restart from their seed, not from an optimizer checkpoint.
Completed fits are reused only after contract and artifact verification.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import resource
import socket
import subprocess
import time
import uuid

import numpy as np
import torch
import scipy
import sklearn

from .detector_cache import digest, immutable_json, iter_cache
from .detector_contract import EXPECTED_PROTOCOL_SHA256
from .features import fit_features, training_sample, transform_features
from .io import atomic_write_json
from .metrics import evaluation_data, evaluate
from .neural_fusion import NeuralFusion, train_neural
from .provenance import _git, sha256_file


ROOT = Path(__file__).resolve().parents[2]


def sources():
    paths = [*sorted((ROOT / "src/thesis_experiments").glob("*.py")),
             ROOT / "scripts/successor/fusion_jobs.py", ROOT / "experiments/protocol/v1.json",
             ROOT / "environments/fusion_constraints.txt"]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def memory_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if platform.system() == "Darwin" else 1024)


def create_plan(root, mode, loss="squared_error", selection_record=None):
    root = Path(root)
    if mode not in ("rehearsal", "benchmark", "grid") or loss not in ("cross_entropy", "squared_error"):
        raise ValueError("invalid development plan mode/loss")
    if sha256_file(ROOT / "experiments/protocol/v1.json") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("frozen protocol changed")
    protocol = json.loads((ROOT / "experiments/protocol/v1.json").read_text())
    cases = []
    if mode == "grid":
        if selection_record is None:
            raise ValueError("grid requires the recorded clean loss pilot")
        selected = json.loads(Path(selection_record).read_text())
        if selected["shared_loss"] != loss or selected.get("test_accessed") is not False:
            raise ValueError("loss pilot identity/scope mismatch")
        for widths in protocol["models"]["mlp"]["hidden_widths"]:
            for lr in protocol["models"]["mlp"]["learning_rate"]:
                cases.append({"kind": "mlp", "widths": widths, "learning_rate": lr})
        for depth in protocol["models"]["vqc"]["depth"]:
            for lr in protocol["models"]["vqc"]["learning_rate"]:
                cases.append({"kind": "vqc", "depth": depth, "learning_rate": lr})
    else:
        cases = [{"kind": "mlp", "widths": [16], "learning_rate": 0.001}]
        for depth in ([2] if mode == "rehearsal" else [1, 2, 3, 4]):
            cases.append({"kind": "vqc", "depth": depth, "learning_rate": 0.001})
    for index, case in enumerate(cases):
        case.update({"id": f"{case['kind']}-{index:02d}", "seed": 11,
                     "epochs": 100 if mode == "grid" else 3, "patience": 10, "loss": loss})
    body = {"schema_version": 1, "mode": mode, "scope": "development-only; test forbidden",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
            "source_sha256": sources(), "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "training_cap": 20000, "torch_threads": 2,
            "selection_record_sha256": sha256_file(Path(selection_record)) if selection_record else None,
            "git_commit": _git(ROOT, "rev-parse", "HEAD"), "git_dirty": bool(_git(ROOT, "status", "--porcelain"))}
    plan = {"plan_id": digest(body), "plan": body}
    root.mkdir(parents=True, exist_ok=False)
    for directory in ("manifests", "caches", "models", "metrics", "logs", "plots"):
        (root / directory).mkdir()
    atomic_write_json(root / "PLAN.json", plan)
    atomic_write_json(root / "provenance.json", plan)
    atomic_write_json(root / "PARTIAL.json", {"stage": "planned", "plan_id": plan["plan_id"]})
    return plan


def load_plan(root):
    result = json.loads((Path(root) / "PLAN.json").read_text())
    body = result["plan"]
    if digest(body) != result["plan_id"] or body["source_sha256"] != sources():
        raise ValueError("plan/source drift; create a new run instead of reusing artifacts")
    if body["mode"] not in ("grid", "benchmark", "rehearsal") or body["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("invalid development contract")
    if (np.__version__, scipy.__version__, sklearn.__version__, torch.__version__.split("+")[0]) != (
            "1.26.4", "1.13.1", "1.5.2", "2.4.1"):
        raise ValueError("use the verified pinned fusion environment")
    for case in body["cases"]:
        if not re.fullmatch(r"(?:mlp|vqc)-\d\d", case["id"]) or case["kind"] not in ("mlp", "vqc"):
            raise ValueError("unsafe task identifier/type")
        if case["epochs"] != (100 if body["mode"] == "grid" else 3):
            raise ValueError("unexpected training limit")
    return result


def case_at(plan, index):
    if not isinstance(index, int) or not 0 <= index < len(plan["plan"]["cases"]):
        raise ValueError("task index outside plan")
    return plan["plan"]["cases"][index]


@contextmanager
def task_lock(path, plan_id):
    path = Path(path)
    try:
        path.mkdir()
    except FileExistsError:
        raise RuntimeError(f"task locked; inspect {path / 'owner.json'} before explicit recovery")
    atomic_write_json(path / "owner.json", {"pid": os.getpid(), "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "plan_id": plan_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        yield
    finally:
        (path / "owner.json").unlink(missing_ok=True)
        path.rmdir()


def verify_files(root, marker):
    root = Path(root).resolve()
    for relative, expected in marker["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"artifact corruption/path mismatch: {relative}")


def unlock_inactive(path, plan_id):
    """Explicit recovery only after proving the recorded owner is no longer active."""
    path = Path(path)
    recovery = path.with_name(path.name + ".recovery")
    recovery.mkdir()
    try:
        original = (path / "owner.json").read_bytes()
        owner = json.loads(original)
        if owner["plan_id"] != plan_id:
            raise ValueError("lock belongs to another plan")
        if owner["host"] == socket.gethostname():
            try:
                os.kill(owner["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("lock owner is still active")
        elif owner.get("slurm_job"):
            status = subprocess.run(["sacct", "-j", str(owner["slurm_job"]), "--format=State", "--noheader", "--parsable2"],
                                    capture_output=True, text=True, check=True)
            states = [line.strip().split("|")[0].split()[0].rstrip("+") for line in status.stdout.splitlines() if line.strip()]
            terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
            if not states or not all(state in terminal for state in states):
                raise RuntimeError("cannot prove remote lock owner inactive")
        else:
            raise RuntimeError("unknown remote lock owner; inspect manually")
        if (path / "owner.json").read_bytes() != original:
            raise RuntimeError("lock owner changed during recovery")
        (path / "owner.json").unlink()
        path.rmdir()
    finally:
        recovery.rmdir()


def prepare(root, cache_root=None):
    root = Path(root)
    plan = load_plan(root)
    destination = root / "caches/prepared"
    with task_lock(root / ".prepare.lock", plan["plan_id"]):
        if (destination / "COMPLETED.json").exists():
            marker = json.loads((destination / "COMPLETED.json").read_text())
            if marker["plan_id"] != plan["plan_id"]:
                raise ValueError("prepared inputs belong to another plan")
            verify_files(destination, marker)
            return "reused"
        started = time.perf_counter()
        staging = root / "caches" / f".prepared-{uuid.uuid4().hex}"
        staging.mkdir()
        if plan["plan"]["mode"] == "rehearsal":
            from .verification import synthetic_cache
            synthetic_cache(staging / "synthetic_train", "train", [f"fit{i}" for i in range(6)])
            synthetic_cache(staging / "synthetic_val", "val", [f"heldout{i}" for i in range(3)])
            train_path, val_path = staging / "synthetic_train", staging / "synthetic_val"
        else:
            if cache_root is None:
                raise ValueError("real development cache root required")
            train_path, val_path = Path(cache_root) / "clean/train", Path(cache_root) / "clean/val"
        training, validation = list(iter_cache(train_path)), list(iter_cache(val_path))
        if any(r["split"] != "train" or r["condition"] != "clean" for r in training):
            raise ValueError("only clean training inputs allowed")
        if any(r["split"] != "val" or r["condition"] != "clean" for r in validation):
            raise ValueError("only clean validation inputs allowed; test forbidden")
        if set(r["sequence"] for r in training) & set(r["sequence"] for r in validation):
            raise ValueError("sequence leakage")
        if plan["plan"]["mode"] != "rehearsal":
            protocol = json.loads((ROOT / "experiments/protocol/v1.json").read_text())
            for rows, split in ((training, "train"), (validation, "val")):
                names = sorted({r["sequence"] for r in rows})
                actual = hashlib_membership(names)
                if actual != protocol["dataset"]["membership_sha256"][split]:
                    raise ValueError("unexpected official sequence membership")
            a, b = [json.loads((p / "CACHE.json").read_text())["context"] for p in (train_path, val_path)]
            for key in ("phase2_manifest_sha256", "development_manifest_sha256", "corruption_specification_sha256"):
                if a[key] != b[key]:
                    raise ValueError(f"cache contract mismatch: {key}")
        fitted = fit_features(training)
        raw, norm, _ = transform_features(training, fitted)
        vr, vn, clipping = transform_features(validation, fitted)
        indices = training_sample(training, plan["plan"]["training_cap"])
        data = evaluation_data(validation)
        arrays = {"train_raw": raw[indices], "train_normalized": norm[indices],
                  "train_labels": np.array([r["paired_presence"] for r in training])[indices],
                  "validation_raw": vr, "validation_normalized": vn,
                  **{f"eval_{k}": v for k, v in data.items() if k != "attributes"}}
        np.savez_compressed(staging / "data.npz", **arrays)
        metadata = {"plan_id": plan["plan_id"], "features": fitted, "validation_clipping": clipping,
                    "train_keys": [training[i]["key"] for i in indices],
                    "validation_attributes": data["attributes"], "preparation_seconds": time.perf_counter() - started,
                    "max_rss_mib": memory_mib(),
                    "input_context_ids": [json.loads((p / "CACHE.json").read_text())["context_id"] for p in (train_path, val_path)]}
        atomic_write_json(staging / "data.json", metadata)
        marker = {"plan_id": plan["plan_id"], "files": {name: sha256_file(staging / name) for name in ("data.npz", "data.json")}}
        load_plan(root)  # source must remain unchanged throughout preparation
        atomic_write_json(staging / "COMPLETED.json", marker)
        staging.rename(destination)
    return "prepared"


def hashlib_membership(names):
    import hashlib
    return hashlib.sha256("".join(name + "\n" for name in names).encode()).hexdigest()


def inputs(root, plan):
    folder = Path(root) / "caches/prepared"
    marker = json.loads((folder / "COMPLETED.json").read_text())
    if marker["plan_id"] != plan["plan_id"]:
        raise ValueError("input/plan identity mismatch")
    verify_files(folder, marker)
    with np.load(folder / "data.npz", allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    metadata = json.loads((folder / "data.json").read_text())
    data = {k[5:]: v for k, v in values.items() if k.startswith("eval_")}
    data["attributes"] = metadata["validation_attributes"]
    return values, data, digest(marker)


def completed_case(root, index):
    root = Path(root)
    plan = load_plan(root)
    case = case_at(plan, index)
    folder = root / "models" / case["id"]
    marker = json.loads((folder / "COMPLETED.json").read_text())
    expected = {"plan_id": plan["plan_id"], "case": case,
                "inputs_id": digest(json.loads((root / "caches/prepared/COMPLETED.json").read_text()))}
    if marker["contract"] != expected:
        raise ValueError("completed task contract differs")
    if marker["result"] not in marker["files"]:
        raise ValueError("result is not bound to the output manifest")
    verify_files(root / "caches/prepared", json.loads((root / "caches/prepared/COMPLETED.json").read_text()))
    verify_files(folder, marker)
    return marker, folder


def run_case(root, index, fail_after_checkpoint=False):
    root = Path(root)
    plan = load_plan(root)
    case = case_at(plan, index)
    if fail_after_checkpoint and plan["plan"]["mode"] != "rehearsal":
        raise ValueError("failure injection is restricted to synthetic rehearsal")
    folder = root / "models" / case["id"]
    folder.mkdir(exist_ok=True)
    with task_lock(folder / ".lock", plan["plan_id"]):
        if (folder / "COMPLETED.json").exists():
            completed_case(root, index)
            return "reused"
        values, data, input_id = inputs(root, plan)
        contract = {"plan_id": plan["plan_id"], "case": case, "inputs_id": input_id}
        immutable_json(folder / "CONTRACT.json", contract)
        attempt = folder / "attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        atomic_write_json(folder / "PARTIAL.json", {"contract": contract, "attempt": str(attempt.relative_to(folder))})
        started = time.perf_counter()
        try:
            torch.set_num_threads(plan["plan"]["torch_threads"])
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(case["seed"])
            model = NeuralFusion(case["kind"], widths=case.get("widths", [16]), depth=case.get("depth", 2))
            info = train_neural(model, values["train_normalized"], values["train_labels"],
                values["validation_normalized"], data, epochs=case["epochs"], patience=case["patience"],
                learning_rate=case["learning_rate"], seed=case["seed"], loss_kind=case["loss"])
            model.save(attempt / "model.pt")
            if fail_after_checkpoint:
                raise RuntimeError("Injected rehearsal interruption after checkpoint, before completion")
            p, u = model.predict(values["validation_normalized"])
            np.savez_compressed(attempt / "predictions.npz", probability=p, uncertainty=u)
            result = {"case": case, "plan_id": plan["plan_id"], "input_id": input_id,
                      "training": info, "validation": evaluate(data, p, u), "total_seconds": time.perf_counter() - started,
                      "max_rss_mib": memory_mib(), "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                      "torch": torch.__version__, "numpy": np.__version__, "scope": plan["plan"]["mode"]}
            atomic_write_json(attempt / "result.json", result)
            files = [attempt / name for name in ("model.pt", "predictions.npz", "result.json")]
            load_plan(root)
            atomic_write_json(folder / "COMPLETED.json", {"contract": contract,
                "result": str((attempt / "result.json").relative_to(folder)),
                "files": {str(p.relative_to(folder)): sha256_file(p) for p in files}})
            (folder / "PARTIAL.json").unlink()
        except BaseException as error:
            atomic_write_json(attempt / "FAILED.json", {"error": str(error), "seconds": time.perf_counter() - started,
                "restart_policy": "New attempt from the same seed; incomplete checkpoint is never accepted"})
            raise
    return "completed"


def aggregate(root):
    root = Path(root)
    plan = load_plan(root)
    with task_lock(root / ".aggregate.lock", plan["plan_id"]):
        results = []
        for index in range(len(plan["plan"]["cases"])):
            marker, folder = completed_case(root, index)
            results.append(json.loads((folder / marker["result"]).read_text()))
        selected = {}
        if plan["plan"]["mode"] == "grid":
            for kind in ("mlp", "vqc"):
                eligible = [r for r in results if r["case"]["kind"] == kind]
                best = max(eligible, key=lambda r: (r["validation"]["edtc_sequence_mean"], -r["validation"]["brier"]))
                selected[kind] = best["case"]
        result = {"plan_id": plan["plan_id"], "mode": plan["plan"]["mode"], "cases": results,
                  "selected_configurations": selected, "test_accessed": False,
                  "restart_policy": "Incomplete fits restart deterministically; completed verified fits are reused"}
        if (root / "COMPLETED.json").exists():
            marker = json.loads((root / "COMPLETED.json").read_text())
            verify_files(root, marker)
            return "reused"
        atomic_write_json(root / "metrics/aggregate.json", result)
        files = [root / "metrics/aggregate.json", root / "PLAN.json", root / "provenance.json",
                 *(root / "caches/prepared" / name for name in ("data.npz", "data.json", "COMPLETED.json"))]
        for index in range(len(plan["plan"]["cases"])):
            marker, folder = completed_case(root, index)
            files.extend(folder / name for name in marker["files"])
            files.extend([folder / "COMPLETED.json", folder / "CONTRACT.json"])
        atomic_write_json(root / "COMPLETED.json", {"plan_id": plan["plan_id"],
            "files": {str(p.relative_to(root)): sha256_file(p) for p in files}})
        (root / "PARTIAL.json").unlink(missing_ok=True)
    return "aggregated"
