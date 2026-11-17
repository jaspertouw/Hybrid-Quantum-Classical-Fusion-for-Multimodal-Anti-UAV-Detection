#!/usr/bin/env python3
"""Short, synthetic-only environment and simulator verification; no dataset reads."""
import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

from src.thesis_experiments.io import atomic_write_json
from src.thesis_experiments.provenance import sha256_file
from src.thesis_experiments.verification import synthetic_run, verify_quantum_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output / "PARTIAL.json", {"status": "running"})
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        a, p = synthetic_run(args.output / "first")
        b, q = synthetic_run(args.output / "repeat")
        if a != b:
            raise AssertionError("repeated synthetic metrics or histories differ")
        for name in p:
            np.testing.assert_allclose(p[name], q[name], rtol=0, atol=1e-12)
        reference = verify_quantum_reference()
    if any(name.startswith("lib.models.yolo_evidential") for name in sys.modules):
        raise AssertionError("synthetic smoke imported inherited detector")
    report = {"scope": "synthetic only; no real training/validation/test data accessed",
              "python": platform.python_version(), "platform": platform.platform(),
              "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "simulator": "exact Torch complex128 statevector",
              "quantum_reference": reference, "repeated_models": list(p),
              "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in
                  [Path(__file__).resolve(), *sorted((ROOT / "src/thesis_experiments").glob("*.py"))]}}
    atomic_write_json(args.output / "report.json", report)
    files = [path for path in args.output.rglob("*") if path.is_file() and path.name not in ("PARTIAL.json", "COMPLETED.json")]
    atomic_write_json(args.output / "COMPLETED.json", {"files": {str(path.relative_to(args.output)): sha256_file(path) for path in files}})
    (args.output / "PARTIAL.json").unlink()
    print(f"FUSION ENVIRONMENT SMOKE PASSED: {args.output}", flush=True)


if __name__ == "__main__":
    main()
