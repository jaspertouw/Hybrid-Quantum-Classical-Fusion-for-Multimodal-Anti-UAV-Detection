#!/usr/bin/env python3
"""Create validated train/validation YOLO data from Anti-UAV-RGBT videos."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2


SPLIT_COUNTS = {"train": 160, "val": 67}
MODALITIES = {
    "thermal": ("infrared.mp4", "infrared.json"),
    "rgb": ("visible.mp4", "visible.json"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_sequence(task: tuple[str, str, str, str]) -> dict:
    dataset_root_s, output_root_s, split, sequence = task
    dataset_root, output_root = Path(dataset_root_s), Path(output_root_s)
    source = dataset_root / split / sequence
    results = {"split": split, "sequence": sequence, "modalities": {}}

    for modality, (video_name, annotation_name) in MODALITIES.items():
        video = source / video_name
        annotation = source / annotation_name
        if not video.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"Missing {video} or {annotation}")

        with annotation.open(encoding="utf-8") as handle:
            labels = json.load(handle)
        boxes, exists = labels.get("gt_rect"), labels.get("exist")
        if not isinstance(boxes, list) or not isinstance(exists, list):
            raise ValueError(f"Invalid annotation structure: {annotation}")
        if len(boxes) != len(exists):
            raise ValueError(f"Annotation length mismatch: {annotation}")

        final_dir = output_root / modality / split / sequence
        marker = final_dir / "complete.json"
        video_hash, annotation_hash = sha256(video), sha256(annotation)
        if marker.is_file():
            previous = json.loads(marker.read_text(encoding="utf-8"))
            expected = {
                "video_sha256": video_hash,
                "annotation_sha256": annotation_hash,
                "frames": len(boxes),
            }
            if all(previous.get(key) == value for key, value in expected.items()):
                image_count = sum(1 for _ in (final_dir / "images").glob("*.jpg"))
                label_count = sum(1 for _ in (final_dir / "labels").glob("*.txt"))
                if image_count != len(boxes) or label_count != len(boxes):
                    raise RuntimeError(f"Incomplete derived data found at {final_dir}")
                results["modalities"][modality] = previous
                continue
            raise RuntimeError(f"Stale derived data found at {final_dir}")
        if final_dir.exists():
            raise RuntimeError(f"Incomplete derived data found at {final_dir}")

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = final_dir.with_name(f".{sequence}.tmp-{os.getpid()}")
        if temporary.exists():
            raise RuntimeError(f"Temporary directory already exists: {temporary}")
        images_dir, labels_dir = temporary / "images", temporary / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir()

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            shutil.rmtree(temporary)
            raise RuntimeError(f"Cannot open video: {video}")

        frame_count = positive_count = clipped_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_count >= len(boxes):
                    raise ValueError(f"Video has more frames than annotations: {video}")

                frame_count += 1
                stem = f"{frame_count:06d}"
                if not cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame):
                    raise RuntimeError(f"Failed to write frame {frame_count} from {video}")

                label_path = labels_dir / f"{stem}.txt"
                if not bool(exists[frame_count - 1]):
                    label_path.write_text("", encoding="utf-8")
                    continue

                box = boxes[frame_count - 1]
                if not isinstance(box, list) or len(box) != 4:
                    raise ValueError(f"Invalid box at frame {frame_count}: {annotation}")
                x, y, width, height = map(float, box)
                image_height, image_width = frame.shape[:2]
                x1, y1 = max(0.0, x), max(0.0, y)
                x2 = min(float(image_width), x + width)
                y2 = min(float(image_height), y + height)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Empty box at frame {frame_count}: {annotation}")
                clipped_count += int((x1, y1, x2 - x1, y2 - y1) != (x, y, width, height))
                positive_count += 1
                yolo = (
                    (x1 + x2) / (2.0 * image_width),
                    (y1 + y2) / (2.0 * image_height),
                    (x2 - x1) / image_width,
                    (y2 - y1) / image_height,
                )
                label_path.write_text(
                    "0 " + " ".join(f"{value:.8f}" for value in yolo) + "\n",
                    encoding="utf-8",
                )
        except Exception:
            capture.release()
            shutil.rmtree(temporary)
            raise
        capture.release()

        if frame_count != len(boxes):
            shutil.rmtree(temporary)
            raise ValueError(
                f"Frame/annotation mismatch for {video}: {frame_count} != {len(boxes)}"
            )

        record = {
            "video_sha256": video_hash,
            "annotation_sha256": annotation_hash,
            "frames": frame_count,
            "positive_frames": positive_count,
            "clipped_boxes": clipped_count,
        }
        (temporary / "complete.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final_dir)
        results["modalities"][modality] = record

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--require-test-absent", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.require_test_absent and (dataset_root / "test").exists():
        raise RuntimeError(f"Test-set lock failed: {dataset_root / 'test'} exists")

    tasks = []
    for split, expected_count in SPLIT_COUNTS.items():
        split_dir = dataset_root / split
        sequences = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        if len(sequences) != expected_count:
            raise ValueError(f"Expected {expected_count} {split} sequences, found {len(sequences)}")
        tasks.extend((str(dataset_root), str(output_root), split, sequence) for sequence in sequences)

    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(prepare_sequence, tasks), 1):
            records.append(result)
            print(f"[{index:03d}/{len(tasks)}] {result['split']}/{result['sequence']}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    for modality in MODALITIES:
        for split in SPLIT_COUNTS:
            image_paths = []
            for record in records:
                if record["split"] != split:
                    continue
                sequence = record["sequence"]
                count = record["modalities"][modality]["frames"]
                image_paths.extend(
                    str(output_root / modality / split / sequence / "images" / f"{i:06d}.jpg")
                    for i in range(1, count + 1)
                )
            (output_root / modality).mkdir(parents=True, exist_ok=True)
            (output_root / modality / f"{split}_images.txt").write_text(
                "\n".join(image_paths) + "\n", encoding="utf-8"
            )
        (output_root / modality / "dataset.yaml").write_text(
            f"path: {output_root / modality}\n"
            "train: train_images.txt\n"
            "val: val_images.txt\n"
            "nc: 1\n"
            "names: ['UAV']\n",
            encoding="utf-8",
        )

    manifest = output_root / "preparation_manifest.json"
    manifest.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Prepared {len(tasks)} train/validation sequences; manifest: {manifest}")
    print("DETECTOR DATA PREPARATION PASSED")


if __name__ == "__main__":
    main()
