#!/usr/bin/env python3
"""Plan, prepare, execute and aggregate restart-safe development fusion tasks."""
import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.thesis_experiments.jobs import aggregate, case_at, create_plan, load_plan, prepare, run_case, unlock_inactive


def terminate(signum, _frame):
    raise InterruptedError(f"worker received signal {signum}; incomplete output must not be consumed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--mode", choices=("grid", "benchmark", "rehearsal"), required=True)
    plan.add_argument("--selection-record", type=Path)
    plan.add_argument("--loss", choices=("cross_entropy", "squared_error"), default="squared_error")
    prep = sub.add_parser("prepare")
    prep.add_argument("--cache-root", type=Path)
    worker = sub.add_parser("worker")
    worker.add_argument("--index", type=int, required=True)
    worker.add_argument("--fail-after-checkpoint", action="store_true")
    sub.add_parser("aggregate")
    sub.add_parser("describe")
    unlock = sub.add_parser("unlock")
    unlock.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, terminate)
    if args.command == "plan":
        result = create_plan(args.work_root, args.mode, args.loss, args.selection_record)
        print(f"Plan {result['plan_id']}: {len(result['plan']['cases'])} tasks, mode={args.mode}")
    elif args.command == "prepare":
        print(prepare(args.work_root, args.cache_root))
    elif args.command == "worker":
        print(run_case(args.work_root, args.index, args.fail_after_checkpoint))
    elif args.command == "aggregate":
        print(aggregate(args.work_root))
    elif args.command == "describe":
        print(json.dumps(load_plan(args.work_root), indent=2))
    else:
        plan = load_plan(args.work_root)
        case = case_at(plan, args.index)
        unlock_inactive(args.work_root / "models" / case["id"] / ".lock", plan["plan_id"])
        print("Inactive lock removed; rerun the same task to restart from its seed")


if __name__ == "__main__":
    main()
