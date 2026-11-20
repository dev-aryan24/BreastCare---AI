
from pathlib import Path
import glob

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_CSV = (
    ROOT / "outputs" / "annotation_analysis"
    / "annotation_regions.csv"
)

SELECTED_CSV = (
    ROOT / "outputs" / "annotation_analysis"
    / "selected_images.csv"
)

OUTPUT_CSV = (
    ROOT / "outputs" / "annotation_analysis"
    / "coordinate_validation.csv"
)

TARGET = "malignant (111).png"
IMAGE_SIZE = 224


def find_original_image(filename, selected):

    if "image_path" in selected.columns:
        row = selected[selected["filename"] == filename]

        if not row.empty:
            path = Path(str(row.iloc[0]["image_path"]))

            if not path.is_absolute():
                path = ROOT / path

            if path.exists():
                return path

    matches = list(
        (ROOT / "data" / "raw" / "BUSI").rglob(filename)
    )

    return matches[0] if matches else None


def load_combined_mask(image_path):

    mask_pattern = (
        image_path.stem + "_mask*.png"
    )

    mask_paths = list(
        image_path.parent.glob(mask_pattern)
    )

    if not mask_paths:
        raise FileNotFoundError(
            f"No ground-truth masks found for {image_path.name}"
        )

    combined = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    for path in mask_paths:

        mask = Image.open(path).convert("L")

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        array = np.asarray(mask)

        combined = np.maximum(
            combined,
            (array > 0).astype(np.uint8)
        )

    return combined, mask_paths


def parse_box(row, prefix):

    if prefix == "annotation":
        columns = ["x1", "y1", "x2", "y2"]
    else:
        columns = [
            "control_x1",
            "control_y1",
            "control_x2",
            "control_y2"
        ]

    values = [row[column] for column in columns]

    if any(pd.isna(value) for value in values):
        raise ValueError(f"{prefix} coordinates are incomplete")

    box = tuple(int(float(value)) for value in values)

    x1, y1, x2, y2 = box

    if not (
        0 <= x1 < x2 < IMAGE_SIZE
        and 0 <= y1 < y2 < IMAGE_SIZE
    ):
        raise ValueError(f"Invalid {prefix} coordinates: {box}")

    return box


def box_mask(box):

    x1, y1, x2, y2 = box

    result = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=bool
    )

    result[y1:y2 + 1, x1:x2 + 1] = True

    return result


def calculate_metrics(box, lesion_mask):

    region = box_mask(box)

    region_pixels = int(region.sum())

    lesion_overlap = int(
        np.logical_and(region, lesion_mask > 0).sum()
    )

    percentage = (
        lesion_overlap / region_pixels * 100
        if region_pixels else 0
    )

    return {
        "region_pixels": region_pixels,
        "lesion_overlap_pixels": lesion_overlap,
        "lesion_overlap_percent": round(percentage, 2)
    }


def main():

    annotations = pd.read_csv(ANNOTATION_CSV)
    selected = pd.read_csv(SELECTED_CSV)

    target_rows = annotations[
        annotations["filename"] == TARGET
    ]

    if target_rows.empty:
        raise ValueError(
            f"{TARGET} is missing from annotation_regions.csv"
        )

    row = target_rows.iloc[0]

    if row["annotation_type"] != "visual_annotation":
        raise ValueError(
            "Target image does not have a visual annotation record."
        )

    image_path = find_original_image(TARGET, selected)

    if image_path is None:
        raise FileNotFoundError(
            f"Original image not found: {TARGET}"
        )

    lesion_mask, mask_paths = load_combined_mask(
        image_path
    )

    annotation = parse_box(row, "annotation")
    control = parse_box(row, "control")

    annotation_region = box_mask(annotation)
    control_region = box_mask(control)

    rectangle_overlap = int(
        np.logical_and(
            annotation_region,
            control_region
        ).sum()
    )

    annotation_metrics = calculate_metrics(
        annotation,
        lesion_mask
    )

    control_metrics = calculate_metrics(
        control,
        lesion_mask
    )

    print("\n========== COORDINATE VALIDATION ==========")

    print("Image:", TARGET)
    print("Ground-truth masks:", len(mask_paths))

    print("\nAnnotation coordinates:", annotation)
    print("Control coordinates:", control)

    print("\nRectangle overlap:", rectangle_overlap)

    print("\nAnnotation region:")
    for key, value in annotation_metrics.items():
        print(f"{key}: {value}")

    print("\nControl region:")
    for key, value in control_metrics.items():
        print(f"{key}: {value}")

    rectangle_pass = rectangle_overlap == 0

    annotation_pass = (
        annotation_metrics["lesion_overlap_pixels"] == 0
    )

    control_pass = (
        control_metrics["lesion_overlap_pixels"] == 0
    )

    print("\n========== VALIDATION STATUS ==========")
    print("Rectangles separate:", rectangle_pass)
    print("Annotation avoids lesion:", annotation_pass)
    print("Control avoids lesion:", control_pass)

    overall = (
        rectangle_pass
        and annotation_pass
        and control_pass
    )

    print("Overall geometric validation:", overall)

    report = pd.DataFrame([{
        "filename": TARGET,
        "annotation_overlap_pixels": rectangle_overlap,
        "annotation_lesion_pixels":
            annotation_metrics["lesion_overlap_pixels"],
        "annotation_lesion_percent":
            annotation_metrics["lesion_overlap_percent"],
        "control_lesion_pixels":
            control_metrics["lesion_overlap_pixels"],
        "control_lesion_percent":
            control_metrics["lesion_overlap_percent"],
        "rectangles_separate": rectangle_pass,
        "annotation_avoids_lesion": annotation_pass,
        "control_avoids_lesion": control_pass,
        "overall_geometric_validation": overall
    }])

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nReport saved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
