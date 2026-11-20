
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    ROOT / "data/raw/BUSI/benign/benign (266).png"
)

MASK_PATH = (
    ROOT / "data/raw/BUSI/benign/benign (266)_mask.png"
)

IMAGE_SIZE = (224, 224)

REGIONS = {
    "text": (57, 145, 126, 159),
    "calipers": (46, 10, 140, 56),
}


def load_mask():
    mask = Image.open(MASK_PATH).convert("L")

    mask = mask.resize(
        IMAGE_SIZE,
        Image.Resampling.NEAREST
    )

    return np.asarray(mask) > 0


def validate_region(name, box, mask):
    x1, y1, x2, y2 = box

    print(f"\n{name.upper()}")

    if not (
        0 <= x1 < x2 <= 224
        and 0 <= y1 < y2 <= 224
    ):
        print("STATUS: FAIL - Invalid coordinates")
        return

    region = mask[y1:y2, x1:x2]

    total_pixels = region.size
    lesion_pixels = int(region.sum())

    overlap = (
        lesion_pixels / total_pixels * 100
        if total_pixels else 0
    )

    print("Coordinates:", box)
    print("Region pixels:", total_pixels)
    print("Lesion overlap pixels:", lesion_pixels)
    print(f"Lesion overlap: {overlap:.2f}%")

    if lesion_pixels == 0:
        print("STATUS: PASS - No geometric lesion overlap")
    else:
        print("STATUS: REVIEW - Lesion overlap detected")


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)

    if not MASK_PATH.exists():
        raise FileNotFoundError(MASK_PATH)

    mask = load_mask()

    print("========== BENIGN 266 VALIDATION ==========")
    print("Image:", IMAGE_PATH.name)
    print("Mask shape:", mask.shape)
    print("Total lesion pixels:", int(mask.sum()))

    for name, box in REGIONS.items():
        validate_region(name, box, mask)


if __name__ == "__main__":
    main()
