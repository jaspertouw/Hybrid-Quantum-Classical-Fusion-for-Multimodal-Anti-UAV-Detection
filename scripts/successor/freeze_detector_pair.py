#!/usr/bin/env python3
"""Verify and freeze the selected RGB/thermal detector pair."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, expected_hash: str) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual_hash}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": actual_hash}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def atomic_create(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace frozen artifact: {path}")
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--thermal-clean", type=Path, required=True)
    parser.add_argument("--thermal-stage1", type=Path, required=True)
    parser.add_argument("--thermal-final", type=Path, required=True)
    parser.add_argument("--rgb-clean", type=Path, required=True)
    parser.add_argument("--rgb-stage1", type=Path, required=True)
    parser.add_argument("--rgb-final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = {
        "protocol": "a6eed103f1badf466f91209bf64134d370f5a077b5e1536f18bdb5620be62741",
        "development_manifest": "63d9d419e207b0ca46eb8821660061bdb66ef51cbe7420abba83328e6a3d371f",
        "thermal_clean": "ed69511c0f936d7fc802ce7507d01155a4eead4e4f54dbfe4870c09ddffdc099",
        "thermal_stage1": "170f8a313ac56b8a27090f86f3e83f8b2a85ea6ed579a962569ce76151421c5c",
        "thermal_final": "db1722f0a018624c94d5b5d08f1c82170936987a46150fb3e51698acb7c9f7c6",
        "rgb_clean": "b2cb8ce08ae39f1c053671ffb53aade5ac9931354108cbb629a75ad40510f6e1",
        "rgb_stage1": "957ea841cf570d3df2b4093c9bb6e907a9c1691df49066481810fa7b02ff318c",
        "rgb_final": "0b3a0e0000946811069d4531f601ce3f3294d1fcc76035134d532ea88a9a1792",
    }
    files = {
        name: checked_file(getattr(args, name), digest)
        for name, digest in expected.items()
    }
    repository = args.repository.resolve()
    status = git(repository, "status", "--porcelain")
    if status:
        raise RuntimeError("repository is dirty; refusing to freeze detector provenance")

    manifest = {
        "schema_version": 1,
        "pair_id": "successor-detector-pair-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": git(repository, "rev-parse", "HEAD"), "dirty": False},
        "protocol": files["protocol"],
        "development_manifest": files["development_manifest"],
        "selection": {
            "policy": "minimum five-epoch rolling Stage-2 training loss; full validation eligibility guard",
            "test_split_used": False,
            "training_jobs": {"clean": "28771695", "gaussian": "29011888", "evidential": "29055813"},
            "verification_jobs": {"gaussian": "29055513", "evidential": "29077991"},
        },
        "modalities": {
            "thermal": {
                "clean": files["thermal_clean"],
                "gaussian": files["thermal_stage1"],
                "selected": files["thermal_final"],
                "best_rolling_train_loss": 0.0205,
                "validation": {"map50": 0.986223, "map50_95": 0.620757, "u_mean": 0.687240},
            },
            "rgb": {
                "clean": files["rgb_clean"],
                "gaussian": files["rgb_stage1"],
                "selected": files["rgb_final"],
                "best_rolling_train_loss": 0.0220,
                "validation": {"map50": 0.988910, "map50_95": 0.705235, "u_mean": 0.698798},
            },
        },
        "environment": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "pytorch_cuda": torch.version.cuda,
            "platform": platform.platform(),
            "training_accelerator": "NVIDIA A100-SXM4-40GB",
        },
    }
    content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    atomic_create(args.output, content)
    manifest_hash = hashlib.sha256(content).hexdigest()
    atomic_create(args.output.with_suffix(args.output.suffix + ".sha256"), f"{manifest_hash}  {args.output.name}\n".encode())
    print(f"Detector-pair manifest: {args.output}")
    print(f"Manifest SHA-256: {manifest_hash}")
    print("DETECTOR PAIR FREEZE PASSED")


if __name__ == "__main__":
    main()
