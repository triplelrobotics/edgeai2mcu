import argparse
import json
import random
import shutil
from pathlib import Path


CLASS_NAMES = ("LEFT", "RIGHT", "STRAIGHT")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def image_files(path: Path):
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Output directory already exists: {dst}")
    shutil.copytree(src, dst)


def copy_hard_examples(hard_dir: Path, out_dir: Path, val_ratio: float, seed: int):
    rng = random.Random(seed)
    copied = {"train": {label: 0 for label in CLASS_NAMES}, "val": {label: 0 for label in CLASS_NAMES}}

    for label in CLASS_NAMES:
        files = image_files(hard_dir / label)
        rng.shuffle(files)
        val_count = round(len(files) * val_ratio)
        if len(files) > 1:
            val_count = max(1, val_count)
        else:
            val_count = 0

        val_files = set(files[:val_count])
        for src in files:
            split = "val" if src in val_files else "train"
            dst = out_dir / split / label / f"hard_{src.name}"
            shutil.copy2(src, dst)
            copied[split][label] += 1

    return copied


def split_counts(out_dir: Path, split: str):
    return {label: len(image_files(out_dir / split / label)) for label in CLASS_NAMES}


def oversample_split(out_dir: Path, split: str, seed: int):
    rng = random.Random(seed)
    counts_before = split_counts(out_dir, split)
    target = max(counts_before.values())
    added = {label: 0 for label in CLASS_NAMES}

    for label in CLASS_NAMES:
        class_dir = out_dir / split / label
        files = image_files(class_dir)
        if not files:
            raise RuntimeError(f"No files to oversample in {class_dir}")

        while len(files) + added[label] < target:
            src = rng.choice(files)
            idx = added[label] + 1
            dst = class_dir / f"balance_{idx:04d}_{src.name}"
            shutil.copy2(src, dst)
            added[label] += 1

    return counts_before, split_counts(out_dir, split), added


def write_labels(out_dir: Path) -> None:
    with open(out_dir / "labels.txt", "w", encoding="utf-8") as f:
        for idx, label in enumerate(CLASS_NAMES):
            f.write(f"{idx} {label}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("dataset/line_training_ready"))
    parser.add_argument("--hard-dir", type=Path, default=Path("dataset/hard_examples"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/line_training_ready_hard_right_balanced"))
    parser.add_argument("--hard-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance", action="store_true", help="Oversample train/val classes to equal counts")
    args = parser.parse_args()

    copy_tree(args.base_dir, args.output_dir)
    hard_copied = copy_hard_examples(args.hard_dir, args.output_dir, args.hard_val_ratio, args.seed)

    balance = {}
    if args.balance:
        for split, offset in (("train", 0), ("val", 1000)):
            before, after, added = oversample_split(args.output_dir, split, args.seed + offset)
            balance[split] = {"before": before, "after": after, "added": added}
    else:
        for split in ("train", "val"):
            counts = split_counts(args.output_dir, split)
            balance[split] = {"before": counts, "after": counts, "added": {label: 0 for label in CLASS_NAMES}}

    summary = {
        "base_dir": str(args.base_dir),
        "hard_dir": str(args.hard_dir),
        "output_dir": str(args.output_dir),
        "hard_val_ratio": args.hard_val_ratio,
        "seed": args.seed,
        "hard_copied": hard_copied,
        "balance": balance,
        "test": split_counts(args.output_dir, "test"),
    }
    with open(args.output_dir / "hard_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_labels(args.output_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
