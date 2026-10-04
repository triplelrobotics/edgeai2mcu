import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}
LABELS = ("LEFT", "RIGHT", "STRAIGHT")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(source: Path):
    records = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        relative = path.relative_to(source)
        if len(relative.parts) < 3:
            raise ValueError(f"Unexpected image path: {relative}")

        split, label = relative.parts[:2]
        if split not in SPLIT_PRIORITY or label not in LABELS:
            raise ValueError(f"Unexpected split or label: {relative}")

        records.append(
            {
                "path": path,
                "relative": relative.as_posix(),
                "split": split,
                "label": label,
                "sha256": file_sha256(path),
            }
        )
    return records


def build_clean_dataset(source: Path, output: Path, force: bool):
    if output.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)

    records = discover_images(source)
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)

    conflicts = []
    for digest, group in by_hash.items():
        labels = sorted({item["label"] for item in group})
        if len(labels) > 1:
            conflicts.append(
                {
                    "sha256": digest,
                    "labels": labels,
                    "paths": sorted(item["relative"] for item in group),
                }
            )
    if conflicts:
        raise ValueError(
            "Identical image content has conflicting labels:\n"
            + json.dumps(conflicts, indent=2)
        )

    output.mkdir(parents=True)
    manifest_rows = []
    counts = Counter()
    cross_split_hashes = 0

    for digest, group in sorted(by_hash.items()):
        splits = {item["split"] for item in group}
        if len(splits) > 1:
            cross_split_hashes += 1

        assigned_split = min(splits, key=SPLIT_PRIORITY.__getitem__)
        candidates = [item for item in group if item["split"] == assigned_split]
        canonical = min(candidates, key=lambda item: item["relative"])
        label = canonical["label"]
        suffix = canonical["path"].suffix.lower()
        destination = output / assigned_split / label / f"{digest[:20]}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(canonical["path"], destination)

        counts[(assigned_split, label)] += 1
        manifest_rows.append(
            {
                "sha256": digest,
                "split": assigned_split,
                "label": label,
                "output_path": destination.relative_to(output).as_posix(),
                "source_paths": "|".join(
                    sorted(item["relative"] for item in group)
                ),
                "copies_removed": len(group) - 1,
            }
        )

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sha256",
                "split",
                "label",
                "output_path",
                "source_paths",
                "copies_removed",
            ),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    (output / "labels.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")

    summary = {
        "source": source.as_posix(),
        "assignment_policy": "test, then val, then train",
        "source_image_files": len(records),
        "unique_image_contents": len(by_hash),
        "duplicate_copies_removed": len(records) - len(by_hash),
        "hashes_previously_crossing_splits": cross_split_hashes,
        "label_conflicts": 0,
        "counts": {
            split: {label: counts[(split, label)] for label in LABELS}
            for split in ("train", "val", "test")
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build a content-deduplicated line-follow dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/line_follow_v2_base_plus_hard_right_20260607"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "dataset/line_follow_v3_base_plus_hard_right_dedup_20261003"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = build_clean_dataset(args.source, args.output, args.force)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
