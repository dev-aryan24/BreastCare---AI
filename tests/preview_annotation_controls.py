
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "BUSI" / "malignant"
OUTPUT_DIR = ROOT / "outputs" / "annotation_analysis" / "control_previews"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)

ANNOTATIONS = {
    "malignant (2).png": (0, 160, 32, 176),
    "malignant (53).png": (96, 16, 144, 32),
}


def load_mask(filename):
    stem = Path(filename).stem
    files = sorted(DATA_DIR.glob(f"{stem}_mask*.png"))

    combined = np.zeros((224, 224), dtype=np.uint8)

    for path in files:
        mask = Image.open(path).convert("L")
        mask = mask.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST,
        )
        combined = np.maximum(
            combined,
            (np.asarray(mask) > 0).astype(np.uint8),
        )

    return combined


def find_candidates(mask, annotation, width, height):
    ax1, ay1, ax2, ay2 = annotation
    candidates = []

    for y1 in range(0, 225 - height, 8):
        for x1 in range(0, 225 - width, 8):
            x2, y2 = x1 + width, y1 + height

            separated = (
                x2 <= ax1 - 8 or x1 >= ax2 + 8
                or y2 <= ay1 - 8 or y1 >= ay2 + 8
            )

            if separated and mask[y1:y2, x1:x2].sum() == 0:
                candidates.append((x1, y1, x2, y2))

    return candidates


def main():
    for filename, annotation in ANNOTATIONS.items():
        image = Image.open(DATA_DIR / filename).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        mask = load_mask(filename)

        width = annotation[2] - annotation[0]
        height = annotation[3] - annotation[1]

        candidates = find_candidates(
            mask, annotation, width, height
        )

        # Select candidates spread across the image.
        selected = []
        for target_y in [24, 72, 120, 168, 208]:
            for target_x in [40, 112, 184]:
                valid = [
                    box for box in candidates
                    if abs(box[0] - target_x) <= 24
                    and abs(box[1] - target_y) <= 24
                ]

                if valid:
                    selected.append(
                        min(
                            valid,
                            key=lambda b:
                            abs(b[0] - target_x)
                            + abs(b[1] - target_y)
                        )
                    )

        selected = list(dict.fromkeys(selected))[:12]

        scale = 3
        preview = image.resize(
            (224 * scale, 224 * scale)
        )

        draw = ImageDraw.Draw(preview)

        ax1, ay1, ax2, ay2 = annotation
        draw.rectangle(
            (ax1*scale, ay1*scale, ax2*scale, ay2*scale),
            outline="red",
            width=3,
        )

        for index, box in enumerate(selected, start=1):
            x1, y1, x2, y2 = box

            draw.rectangle(
                (x1*scale, y1*scale, x2*scale, y2*scale),
                outline="lime",
                width=2,
            )

            draw.text(
                (x1*scale, max(0, y1*scale - 15)),
                str(index),
                fill="yellow",
            )

        output = OUTPUT_DIR / (
            Path(filename).stem.replace(" ", "_")
            + "_controls.png"
        )

        preview.save(output)

        print(f"\nImage: {filename}")
        print(f"Annotation: {annotation}")
        print(f"Candidates displayed: {len(selected)}")
        for index, box in enumerate(selected, start=1):
            print(f"{index}: {box}")

        print(f"Preview saved: {output}")


if __name__ == "__main__":
    main()
