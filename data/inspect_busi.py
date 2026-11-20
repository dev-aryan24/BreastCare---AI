
from pathlib import Path
import json
import re

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "BUSI"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ultrasound"
REPORT_DIR = PROJECT_ROOT / "docs" / "research"

CLASSES = ["benign", "malignant", "normal"]
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_class(path):
    for parent in path.parents:
        if parent.name.lower() in CLASSES:
            return parent.name.lower()
    return None


def is_mask(path):
    return bool(
        re.search(r"_mask(?:_\d+)?$", path.stem, re.IGNORECASE)
    )


def original_stem(path):
    return re.sub(
        r"_mask(?:_\d+)?$",
        "",
        path.stem,
        flags=re.IGNORECASE
    )


def validate_image(path):
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            return True, img.size

    except (OSError, UnidentifiedImageError, ValueError):
        return False, None


def inspect_dataset():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    files = [
        p for p in DATASET_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    records = []
    class_images = {name: [] for name in CLASSES}
    class_masks = {name: [] for name in CLASSES}

    for path in files:
        label = get_class(path)

        if label is None:
            continue

        valid, dimensions = validate_image(path)

        record = {
            "filename": path.name,
            "class": label,
            "type": "mask" if is_mask(path) else "image",
            "valid": valid,
            "width": dimensions[0] if dimensions else None,
            "height": dimensions[1] if dimensions else None,
            "path": str(path.relative_to(PROJECT_ROOT))
        }

        records.append(record)

        if is_mask(path):
            class_masks[label].append(path)
        else:
            class_images[label].append(path)

    rows = []
    pairing_report = {}

    for label in CLASSES:
        image_names = {
            p.stem.lower(): p for p in class_images[label]
        }

        mask_names = {
            original_stem(p).lower() for p in class_masks[label]
        }

        valid_images = sum(
            validate_image(p)[0] for p in class_images[label]
        )

        valid_masks = sum(
            validate_image(p)[0] for p in class_masks[label]
        )

        matched = set(image_names) & mask_names
        missing_masks = set(image_names) - mask_names
        orphan_masks = mask_names - set(image_names)

        rows.append({
            "class": label,
            "original_images": len(class_images[label]),
            "mask_files": len(class_masks[label]),
            "valid_images": valid_images,
            "valid_masks": valid_masks,
            "images_with_masks": len(matched),
            "images_without_masks": len(missing_masks),
            "orphan_mask_stems": len(orphan_masks)
        })

        pairing_report[label] = {
            "matched_images": sorted(matched),
            "images_without_masks": sorted(missing_masks),
            "orphan_masks": sorted(orphan_masks)
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(
        OUTPUT_DIR / "dataset_summary.csv",
        index=False
    )

    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / "file_validation_report.csv",
        index=False
    )

    with open(
        REPORT_DIR / "busi_quality_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump({
            "dataset_root": str(DATASET_ROOT),
            "total_supported_files": len(files),
            "class_summary": rows,
            "mask_pairing": pairing_report
        }, file, indent=4)

    print("\n========== BUSI DATASET REPORT ==========\n")
    print(dataframe.to_string(index=False))

    print("\nTotal original images:",
          sum(len(v) for v in class_images.values()))

    print("Total mask files:",
          sum(len(v) for v in class_masks.values()))

    print("Corrupted files:",
          sum(not r["valid"] for r in records))

    print("\nReports saved successfully.")
    print("Location:", OUTPUT_DIR)

    return class_images, class_masks


def show_samples(class_images, class_masks):
    available = [
        label for label in CLASSES
        if class_images[label]
    ]

    if not available:
        print("No images available for visualization.")
        return

    fig, axes = plt.subplots(
        len(available), 3,
        figsize=(12, 4 * len(available)),
        squeeze=False
    )

    for row, label in enumerate(available):
        image_path = class_images[label][0]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{label.title()} - Original")

        matching_masks = [
            p for p in class_masks[label]
            if original_stem(p).lower() == image_path.stem.lower()
        ]

        if matching_masks:
            with Image.open(matching_masks[0]) as mask:
                mask_image = mask.convert("L")

            axes[row, 1].imshow(mask_image, cmap="gray")
            axes[row, 1].set_title("Segmentation Mask")

            axes[row, 2].imshow(image)
            axes[row, 2].imshow(
                mask_image,
                cmap="jet",
                alpha=0.35
            )
            axes[row, 2].set_title("Image + Mask Overlay")
        else:
            for col in [1, 2]:
                axes[row, col].text(
                    0.5, 0.5,
                    "No matching mask",
                    ha="center",
                    va="center"
                )
                axes[row, col].set_title("Mask unavailable")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "busi_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print("\nSample visualization saved:", output_path)

    plt.show()


if __name__ == "__main__":
    images, masks = inspect_dataset()
    show_samples(images, masks)
