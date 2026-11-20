
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "BUSI" / "malignant"

IMAGE_SIZE = (224, 224)

CANDIDATES = {
    "malignant (2).png": [
        {
            "name": "RT1_text",
            "box": (0, 160, 32, 176),
        },
        {
            "name": "measurement_marker",
            "box": (128, 160, 144, 176),
        },
    ],
    "malignant (53).png": [
        {
            "name": "SATELLITE_text",
            "box": (96, 16, 144, 32),
        },
    ],
}


def load_combined_mask(image_name):
    stem = Path(image_name).stem

    mask_files = sorted(
        DATA_DIR.glob(f"{stem}_mask*.png")
    )

    if not mask_files:
        raise FileNotFoundError(
            f"No masks found for {image_name}"
        )

    combined = np.zeros(
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        dtype=np.uint8,
    )

    for mask_path in mask_files:
        mask = Image.open(mask_path).convert("L")

        mask = mask.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST,
        )

        mask_array = np.asarray(mask)

        binary_mask = (mask_array > 0).astype(
            np.uint8
        )

        combined = np.maximum(
            combined,
            binary_mask,
        )

        print(f"  Loaded mask: {mask_path.name}")

    return combined


def validate_region(box, lesion_mask):
    x1, y1, x2, y2 = box

    height, width = lesion_mask.shape

    if not (
        0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
    ):
        raise ValueError(
            f"Invalid rectangle: {box}"
        )

    region = lesion_mask[y1:y2, x1:x2]

    total_pixels = region.size
    lesion_pixels = int(region.sum())

    overlap_percentage = (
        lesion_pixels / total_pixels
    ) * 100

    return {
        "region_pixels": total_pixels,
        "lesion_pixels": lesion_pixels,
        "overlap_percentage": overlap_percentage,
    }


def main():
    print("=" * 65)
    print("ANNOTATION REGION VALIDATION")
    print("=" * 65)

    for image_name, regions in CANDIDATES.items():
        print(f"\nIMAGE: {image_name}")

        image_path = DATA_DIR / image_name

        if not image_path.exists():
            print("ERROR: Original image missing")
            continue

        image = Image.open(image_path).convert("RGB")

        print(f"Original dimensions: {image.size}")

        lesion_mask = load_combined_mask(image_name)

        print(
            f"Combined lesion-mask pixels: "
            f"{int(lesion_mask.sum())}"
        )

        for item in regions:
            result = validate_region(
                item["box"],
                lesion_mask,
            )

            print(f"\nRegion: {item['name']}")
            print(f"Coordinates: {item['box']}")
            print(
                f"Region pixels: "
                f"{result['region_pixels']}"
            )
            print(
                f"Lesion overlap pixels: "
                f"{result['lesion_pixels']}"
            )
            print(
                f"Lesion overlap: "
                f"{result['overlap_percentage']:.2f}%"
            )

            if result["overlap_percentage"] == 0:
                print("Geometric lesion overlap: PASS")
            else:
                print("Geometric lesion overlap: REVIEW")

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
