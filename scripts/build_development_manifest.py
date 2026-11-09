#!/usr/bin/env python3
"""Build the immutable train/validation frame manifest."""

import argparse
from pathlib import Path

from src.thesis_experiments.manifest import build_development_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    summary = build_development_manifest(
        args.dataset_root,
        args.output,
        args.summary,
        derived_root=args.derived_root,
        check_images=args.check_images,
    )
    for split, counts in summary["splits"].items():
        print(
            f"{split}: {counts['sequences']} sequences, {counts['frames']} frames, "
            f"{counts['paired_present']} present, {counts['paired_absent']} absent"
        )
    print(f"Manifest SHA-256: {summary['manifest_sha256']}")
    print("DEVELOPMENT MANIFEST PASSED")


if __name__ == "__main__":
    main()
