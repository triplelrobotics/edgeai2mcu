import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

import cv2


LABELS = ("LEFT", "RIGHT", "STRAIGHT")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_impulse_labels(split_dir: Path):
    labels_path = split_dir / "info.labels"
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data.get("files", []):
        label = item.get("label", {}).get("label")
        rel_path = item.get("path")
        if label not in LABELS or not rel_path:
            continue
        records.append({
            "path": rel_path,
            "name": item.get("name", Path(rel_path).stem),
            "label": label,
            "category": item.get("category", split_dir.name),
        })
    return records


def image_size(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = image.shape[:2]
    return w, h


def stratified_val_split(records, val_ratio: float, seed: int):
    rng = Random(seed)
    by_label = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    train_records = []
    val_records = []
    for label in LABELS:
        items = list(by_label[label])
        rng.shuffle(items)
        val_count = max(1, round(len(items) * val_ratio)) if items else 0
        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:])

    return train_records, val_records


def safe_name(split: str, label: str, index: int, src_name: str):
    suffix = Path(src_name).suffix.lower()
    if suffix not in IMAGE_EXTS:
        suffix = ".jpg"
    return f"{split}_{label}_{index:05d}{suffix}"


def copy_records(records, source_dir: Path, output_dir: Path, split: str, manifest_rows):
    counts = Counter()
    for index, record in enumerate(records, start=1):
        label = record["label"]
        src = source_dir / record["path"]
        if not src.exists():
            raise FileNotFoundError(f"Missing image listed in info.labels: {src}")

        width, height = image_size(src)
        dst_dir = output_dir / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_name = safe_name(split, label, counts[label] + 1, src.name)
        dst = dst_dir / dst_name
        shutil.copy2(src, dst)

        manifest_rows.append({
            "split": split,
            "label": label,
            "width": width,
            "height": height,
            "src": str(src),
            "dst": str(dst),
            "edge_impulse_name": record["name"],
        })
        counts[label] += 1
    return counts


def write_manifest(path: Path, rows):
    fieldnames = ["split", "label", "width", "height", "src", "dst", "edge_impulse_name"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_labels(path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for idx, label in enumerate(LABELS):
            f.write(f"{idx} {label}\n")


def write_summary(path: Path, manifest_rows):
    summary = {
        "labels": list(LABELS),
        "counts": {},
        "sizes": {},
    }
    for row in manifest_rows:
        split = row["split"]
        label = row["label"]
        summary["counts"].setdefault(split, {})
        summary["counts"][split][label] = summary["counts"][split].get(label, 0) + 1
        summary["sizes"].setdefault(split, Counter())
        summary["sizes"][split][f"{row['width']}x{row['height']}"] += 1

    summary["sizes"] = {
        split: dict(counter)
        for split, counter in summary["sizes"].items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def prepare_dataset(source_root: Path, output_root: Path, val_ratio: float, seed: int):
    training_dir = source_root / "training"
    testing_dir = source_root / "testing"
    if not training_dir.exists() or not testing_dir.exists():
        raise FileNotFoundError(f"Expected training/ and testing/ under {source_root}")

    trainval_records = load_impulse_labels(training_dir)
    test_records = load_impulse_labels(testing_dir)
    train_records, val_records = stratified_val_split(trainval_records, val_ratio, seed)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    counts = {
        "train": copy_records(train_records, training_dir, output_root, "train", manifest_rows),
        "val": copy_records(val_records, training_dir, output_root, "val", manifest_rows),
        "test": copy_records(test_records, testing_dir, output_root, "test", manifest_rows),
    }

    write_manifest(output_root / "manifest.csv", manifest_rows)
    write_labels(output_root / "labels.txt")
    write_summary(output_root / "summary.json", manifest_rows)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/exported_from_impulse"),
        help="Edge Impulse export directory containing training/ and testing/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/line_training_ready"),
        help="Output directory for class-folder train/val/test data.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = prepare_dataset(args.source, args.output, args.val_ratio, args.seed)
    print(f"Prepared dataset: {args.output}")
    for split in ("train", "val", "test"):
        print(split, dict(counts[split]))


if __name__ == "__main__":
    main()
