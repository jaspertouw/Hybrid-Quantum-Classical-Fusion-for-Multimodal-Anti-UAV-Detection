"""Validated train/validation frame-manifest construction."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from .io import atomic_write_json


SCHEMA_VERSION = 1
EXPECTED_SEQUENCE_COUNTS = {"train": 160, "val": 67}
ALLOWED_SPLITS = frozenset(EXPECTED_SEQUENCE_COUNTS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation(path: Path) -> tuple[list, list]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    exists, boxes = value.get("exist"), value.get("gt_rect")
    if not isinstance(exists, list) or not isinstance(boxes, list):
        raise ValueError(f"invalid annotation structure: {path}")
    if len(exists) != len(boxes):
        raise ValueError(f"annotation length mismatch: {path}")
    return exists, boxes


def _box(box: object, *, path: Path, frame: int) -> tuple[Optional[list[float]], bool]:
    if box == []:
        return None, False
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"invalid box in {path} at frame {frame}")
    result = [float(value) for value in box]
    if not all(value == value and abs(value) != float("inf") for value in result):
        raise ValueError(f"non-finite box in {path} at frame {frame}")
    return result, result[2] > 0.0 and result[3] > 0.0


def _attributes(dataset_root: Path, split: str) -> dict[str, list[str]]:
    path = dataset_root / "label_new" / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"invalid attribute mapping: {path}")
    return value


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _check_readable(path: Path) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required when check_images=True") from error
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError(f"unreadable image: {path}")


def build_development_manifest(
    dataset_root: Path,
    output_path: Path,
    summary_path: Path,
    *,
    derived_root: Optional[Path] = None,
    splits: Iterable[str] = ("train", "val"),
    check_images: bool = False,
) -> dict:
    """Build an atomic JSONL manifest without reading the test split."""
    dataset_root = dataset_root.resolve()
    selected_splits = tuple(splits)
    if not selected_splits or len(set(selected_splits)) != len(selected_splits):
        raise ValueError("splits must be non-empty and unique")
    forbidden = set(selected_splits) - ALLOWED_SPLITS
    if forbidden:
        raise ValueError(f"development manifest forbids splits: {sorted(forbidden)}")
    if check_images and derived_root is None:
        raise ValueError("derived_root is required when check_images=True")
    derived_root = derived_root.resolve() if derived_root else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=str(output_path.parent)
    )
    digest = hashlib.sha256()
    seen_keys: set[str] = set()
    source_hashes: dict[str, str] = {}
    split_summaries: dict[str, dict] = {}

    try:
        with os.fdopen(descriptor, "wb") as output:
            for split in selected_splits:
                split_dir = dataset_root / split
                sequences = sorted(
                    path for path in split_dir.iterdir()
                    if path.is_dir()
                    and (path / "infrared.json").is_file()
                    and (path / "visible.json").is_file()
                )
                expected = EXPECTED_SEQUENCE_COUNTS[split]
                if len(sequences) != expected:
                    raise ValueError(
                        f"expected {expected} {split} sequences, found {len(sequences)}"
                    )
                attributes = _attributes(dataset_root, split)
                if set(attributes) != {path.name for path in sequences}:
                    raise ValueError(f"attribute membership mismatch for {split}")

                counts = Counter(sequences=len(sequences))
                attribute_counts: Counter[str] = Counter()
                for sequence_dir in sequences:
                    sequence = sequence_dir.name
                    ir_path, rgb_path = sequence_dir / "infrared.json", sequence_dir / "visible.json"
                    ir_exists, ir_boxes = _annotation(ir_path)
                    rgb_exists, rgb_boxes = _annotation(rgb_path)
                    lengths = {len(ir_exists), len(rgb_exists)}
                    if len(lengths) != 1:
                        raise ValueError(f"sensor frame-count mismatch: {split}/{sequence}")
                    tags = attributes[sequence]
                    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                        raise ValueError(f"invalid attributes: {split}/{sequence}")
                    attribute_counts.update(tags)
                    source_hashes[_relative(ir_path, dataset_root)] = _sha256(ir_path)
                    source_hashes[_relative(rgb_path, dataset_root)] = _sha256(rgb_path)

                    for index in range(len(ir_exists)):
                        frame = index + 1
                        ir_box, ir_box_valid = _box(ir_boxes[index], path=ir_path, frame=frame)
                        rgb_box, rgb_box_valid = _box(rgb_boxes[index], path=rgb_path, frame=frame)
                        ir_present, rgb_present = bool(ir_exists[index]), bool(rgb_exists[index])
                        key = f"{split}/{sequence}/{frame:06d}"
                        if key in seen_keys:
                            raise ValueError(f"duplicate frame key: {key}")
                        seen_keys.add(key)

                        record = {
                            "schema_version": SCHEMA_VERSION,
                            "key": key,
                            "split": split,
                            "sequence": sequence,
                            "frame": frame,
                            "paired_presence": ir_present or rgb_present,
                            "attributes": tags,
                            "thermal": {
                                "present": ir_present,
                                "box_xywh": ir_box,
                                "box_valid": ir_box_valid,
                            },
                            "rgb": {
                                "present": rgb_present,
                                "box_xywh": rgb_box,
                                "box_valid": rgb_box_valid,
                            },
                        }
                        if derived_root is not None:
                            thermal_image = (
                                derived_root / "thermal" / split / sequence / "images" / f"{frame:06d}.jpg"
                            )
                            rgb_image = (
                                derived_root / "rgb" / split / sequence / "images" / f"{frame:06d}.jpg"
                            )
                            if not thermal_image.is_file() or not rgb_image.is_file():
                                raise FileNotFoundError(f"missing paired images for {key}")
                            if check_images:
                                _check_readable(thermal_image)
                                _check_readable(rgb_image)
                            record["thermal"]["image"] = _relative(thermal_image, derived_root)
                            record["rgb"]["image"] = _relative(rgb_image, derived_root)

                        counts["frames"] += 1
                        counts["paired_present"] += int(record["paired_presence"])
                        counts["paired_absent"] += int(not record["paired_presence"])
                        counts["thermal_present"] += int(ir_present)
                        counts["rgb_present"] += int(rgb_present)
                        counts["thermal_invalid_positive_box"] += int(ir_present and not ir_box_valid)
                        counts["rgb_invalid_positive_box"] += int(rgb_present and not rgb_box_valid)

                        line = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
                        output.write(line)
                        digest.update(line)
                split_summaries[split] = {
                    **dict(sorted(counts.items())),
                    "attribute_counts": dict(sorted(attribute_counts.items())),
                }
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(output_path.resolve()),
        "manifest_sha256": digest.hexdigest(),
        "dataset_root": str(dataset_root),
        "derived_root": str(derived_root) if derived_root else None,
        "splits": split_summaries,
        "source_annotation_sha256": dict(sorted(source_hashes.items())),
    }
    atomic_write_json(summary_path, summary)
    return summary
