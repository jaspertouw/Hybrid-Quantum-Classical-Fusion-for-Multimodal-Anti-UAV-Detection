#!/usr/bin/env python3
"""Validate a frozen experiment protocol and, optionally, its dataset split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_PROTOCOL = Path("experiments/protocol/v1.json")


def load_protocol(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol["protocol"]["status"] != "frozen":
        raise ValueError("protocol status must be 'frozen'")
    if protocol["dataset"]["paired_presence"] != "ir_exist OR rgb_exist":
        raise ValueError("paired-presence rule changed")
    if protocol["detector_preprocessing"]["candidate_score_minimum"] != 0.25:
        raise ValueError("candidate threshold changed")
    if protocol["detector_preprocessing"]["nms"]["iou_threshold"] != 0.45:
        raise ValueError("NMS threshold changed")
    if protocol["features"]["box_selection"]["tie_winner"] != "thermal":
        raise ValueError("box tie-break rule changed")
    if len(protocol["features"]["order"]) != 8:
        raise ValueError("fusion input must contain exactly eight ordered features")
    if protocol["training"]["final_seeds"] != [11, 22, 33, 44, 55]:
        raise ValueError("final seed set changed")
    if protocol["corruptions"]["master_noise_seed"] != 2026:
        raise ValueError("corruption master seed changed")
    if protocol["evaluation"]["bootstrap_replicates"] != 2000:
        raise ValueError("bootstrap replicate count changed")
    return protocol


def canonical_membership(split_dir: Path) -> tuple[int, str]:
    names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    digest = hashlib.sha256("".join(name + "\n" for name in names).encode("utf-8")).hexdigest()
    return len(names), digest


def validate_dataset(protocol: dict, dataset_root: Path) -> None:
    counts = protocol["dataset"]["expected_sequence_counts"]
    hashes = protocol["dataset"]["membership_sha256"]
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            raise ValueError(f"missing dataset split: {split_dir}")
        actual_count, actual_hash = canonical_membership(split_dir)
        if actual_count != counts[split]:
            raise ValueError(f"{split}: expected {counts[split]} sequences, found {actual_count}")
        if actual_hash != hashes[split]:
            raise ValueError(f"{split}: sequence-membership SHA-256 does not match protocol")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dataset-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    print(f"Protocol {protocol['protocol']['version']} is structurally valid and frozen.")
    if args.dataset_root:
        validate_dataset(protocol, args.dataset_root)
        print("Dataset split counts and membership hashes match the frozen protocol.")


if __name__ == "__main__":
    main()
