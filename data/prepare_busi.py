
import random
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "BUSI"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ultrasound"

CLASSES = ["benign", "malignant", "normal"]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def find_class_directories():
    folders = [DATASET_ROOT] + [
        p for p in DATASET_ROOT.rglob("*") if p.is_dir()
    ]

    candidates = []

    for folder in folders:
        if not folder.exists():
            continue

        children = {
            item.name.lower(): item
            for item in folder.iterdir()
            if item.is_dir()
        }

        if all(label in children for label in CLASSES):
            candidates.append({
                label: children[label] for label in CLASSES
            })

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one BUSI dataset root, found {len(candidates)}. "
            f"Dataset location: {DATASET_ROOT}"
        )

    return candidates[0]


def get_masks(image_path):
    masks = []

    for path in image_path.parent.glob(image_path.stem + "_mask*"):
        if path.suffix.lower() in EXTENSIONS:
            masks.append(path)

    return sorted(masks)


def prepare_dataset():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = find_class_directories()
    records = []
    summary = []

    for label in CLASSES:
        images = sorted(
            p for p in class_dirs[label].iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
            and "_mask" not in p.stem.lower()
        )

        random.Random(SEED + CLASSES.index(label)).shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        splits = {
            "train": images[:train_end],
            "validation": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, split_images in splits.items():
            for image in split_images:
                masks = get_masks(image)

                records.append({
                    "image_path": str(image.resolve()),
                    "class": label,
                    "split": split,
                    "mask_paths": ";".join(
                        str(mask.resolve()) for mask in masks
                    ),
                    "mask_count": len(masks)
                })

        summary.append({
            "class": label,
            "total": total,
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test": len(splits["test"])
        })

    manifest_path = OUTPUT_DIR / "busi_manifest.csv"
    summary_path = OUTPUT_DIR / "split_summary.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print("\n========== BUSI SPLIT SUMMARY ==========")
    for row in summary:
        print(
            f"{row['class']:10} | "
            f"Train: {row['train']:3} | "
            f"Validation: {row['validation']:3} | "
            f"Test: {row['test']:3} | "
            f"Total: {row['total']:3}"
        )

    print(f"\nTotal images: {len(records)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print("\nDataset preparation completed.")


if __name__ == "__main__":
    prepare_dataset()
