
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data/processed/ultrasound/busi_manifest.csv"
SELECTED = ROOT / "outputs/annotation_analysis/selected_images.csv"

OUTPUT = ROOT / "outputs/annotation_analysis/full_test_audit"

GRID_DIR = OUTPUT / "coordinate_grids"
CONTACT_DIR = OUTPUT / "contact_sheets"

IMAGE_SIZE = 224
GRID_STEP = 16
SHEET_SIZE = 20


def get_filename(path):
    return Path(str(path)).name


def create_coordinate_grid(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    draw = ImageDraw.Draw(image)

    for x in range(0, IMAGE_SIZE + 1, GRID_STEP):
        draw.line(
            (x, 0, x, IMAGE_SIZE),
            fill=(0, 255, 0),
            width=1
        )

        if x < IMAGE_SIZE:
            draw.text(
                (x + 2, 2),
                str(x),
                fill=(255, 255, 0)
            )

    for y in range(0, IMAGE_SIZE + 1, GRID_STEP):
        draw.line(
            (0, y, IMAGE_SIZE, y),
            fill=(0, 255, 0),
            width=1
        )

        if y < IMAGE_SIZE:
            draw.text(
                (2, y + 2),
                str(y),
                fill=(255, 255, 0)
            )

    image.save(output_path)


def create_contact_sheets(records):
    for start in range(0, len(records), SHEET_SIZE):
        batch = records.iloc[start:start + SHEET_SIZE]

        fig, axes = plt.subplots(
            4, 5,
            figsize=(18, 15)
        )

        axes = axes.flatten()

        for ax, (_, record) in zip(axes, batch.iterrows()):
            filename = record["filename"]
            class_name = record["class"]

            grid_path = GRID_DIR / f"{Path(filename).stem}_grid.png"

            image = Image.open(grid_path).convert("RGB")

            ax.imshow(image)
            ax.set_title(
                f"{class_name.upper()}\n{filename}",
                fontsize=8
            )
            ax.axis("off")

        for ax in axes[len(batch):]:
            ax.axis("off")

        page = start // SHEET_SIZE + 1

        fig.suptitle(
            f"BUSI Test Annotation Audit - Page {page}",
            fontsize=15
        )

        plt.tight_layout()

        fig.savefig(
            CONTACT_DIR / f"audit_page_{page}.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)


def main():
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    selected = pd.read_csv(SELECTED)

    test = manifest[manifest["split"] == "test"].copy()

    test["filename"] = test["image_path"].apply(get_filename)

    selected_names = set(
        selected["filename"].astype(str)
    )

    remaining = test[
        ~test["filename"].isin(selected_names)
    ].copy()

    print("========== TEST AUDIT GENERATION ==========")
    print("Total test images:", len(test))
    print("Previously selected:", len(selected_names))
    print("Remaining images:", len(remaining))

    if len(test) != 120:
        raise ValueError("Expected 120 test images.")

    if len(remaining) != 100:
        raise ValueError(
            f"Expected 100 remaining images, got {len(remaining)}"
        )

    failures = []

    for _, record in remaining.iterrows():
        filename = record["filename"]
        image_path = Path(record["image_path"])

        output_path = (
            GRID_DIR /
            f"{Path(filename).stem}_grid.png"
        )

        try:
            if not image_path.exists():
                raise FileNotFoundError(image_path)

            create_coordinate_grid(
                image_path,
                output_path
            )

        except Exception as error:
            failures.append((filename, str(error)))

    if failures:
        print("\nFAILED IMAGES:")
        for filename, error in failures:
            print(filename, ":", error)

        raise RuntimeError(
            f"{len(failures)} images failed."
        )

    create_contact_sheets(remaining)

    audit_template = pd.DataFrame({
        "filename": remaining["filename"],
        "annotation_id": "",
        "annotation_type": "",
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "control_x1": "",
        "control_y1": "",
        "control_x2": "",
        "control_y2": "",
        "visual_verified": False,
        "notes": ""
    })

    template_path = OUTPUT / "annotation_audit_template.csv"

    audit_template.to_csv(
        template_path,
        index=False
    )

    print("\n========== GENERATION COMPLETE ==========")
    print("Coordinate grids:", len(list(GRID_DIR.glob("*_grid.png"))))
    print("Contact sheets:", len(list(CONTACT_DIR.glob("*.png"))))
    print("Audit template:", template_path)
    print("Failed images:", len(failures))

    print("\nCLASS DISTRIBUTION:")
    print(remaining["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
