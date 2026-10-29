"""Create and inspect portable successor experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_experiment_config
from .run import RunLayout


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=Path("experiments/config/default.json"))
    result.add_argument("--dataset-root", type=Path)
    result.add_argument("--checkpoint-root", type=Path)
    result.add_argument("--output-root", type=Path)
    subcommands = result.add_subparsers(dest="command", required=True)

    initialize = subcommands.add_parser("init", help="initialize a partial run")
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--input", type=Path, action="append", default=[])

    complete = subcommands.add_parser("complete", help="mark a run complete after outputs exist")
    complete.add_argument("--run-id", required=True)
    complete.add_argument("--required-output", type=Path, action="append", required=True)

    status = subcommands.add_parser("status", help="report run state")
    status.add_argument("--run-id", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_experiment_config(
        args.config,
        dataset_root=args.dataset_root,
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_root,
    )
    run = RunLayout(config, args.run_id)
    if args.command == "init":
        run.initialize(args.input)
        print(f"Initialized partial run: {run.root}")
    elif args.command == "complete":
        outputs = [path if path.is_absolute() else run.root / path for path in args.required_output]
        run.mark_complete(outputs)
        print(f"Completed run: {run.root}")
    else:
        print(f"{run.run_id}: {run.state()} ({run.root})")


if __name__ == "__main__":
    main()
