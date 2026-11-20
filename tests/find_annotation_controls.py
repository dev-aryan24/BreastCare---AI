
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "BUSI" / "malignant"

IMAGE_SIZE = (224, 224)

ANNOTATIONS = {
    "malignant (2).png": {
        "name": "RT1_text",
        "box": (0, 160, 32, 176),
    },
    "malignant (53).png": {
        "name": "SATELLITE_text",
        "box": (96, 16, 144, 32),
    },
}


def load_mask(filename):
    stem = Path(filename).stem
    masks = sorted(DATA_DIR.glob(f"{stem}_mask*.png"))

    combined = np.zeros((224, 224), dtype=np.uint8)

    for path in masks:
        image = Image.open(path).convert("L")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST,
        )
        combined = np.maximum(
            combined,
            (np.asarray(image) > 0).astype(np.uint8),
        )

    return combined


def find_controls(mask, annotation_box, width, height):
    ax1, ay1, ax2, ay2 = annotation_box
    candidates = []

    for y1 in range(0, 225 - height, 8):
        for x1 in range(0, 225 - width, 8):
            x2 = x1 + width
            y2 = y1 + height

            # Exclude annotation rectangle and nearby area.
            separated = (
                x2 <= ax1 - 8
                or x1 >= ax2 + 8
                or y2 <= ay1 - 8
                or y1 >= ay2 + 8
            )

            if not separated:
                continue

            region = mask[y1:y2, x1:x2]

            if region.sum() == 0:
                candidates.append((x1, y1, x2, y2))

    return candidates


def main():
    for filename, item in ANNOTATIONS.items():
        print("\n" + "=" * 60)
        print(filename)
        print("Annotation:", item["box"])

        mask = load_mask(filename)

        x1, y1, x2, y2 = item["box"]
        width = x2 - x1
        height = y2 - y1

        candidates = find_controls(
            mask,
            item["box"],
            width,
            height,
        )

        print("Zero-overlap control candidates:", len(candidates))
        print("First 15 candidates:")

        for box in candidates[:15]:
            print(box)


if __name__ == "__main__":
    main()
