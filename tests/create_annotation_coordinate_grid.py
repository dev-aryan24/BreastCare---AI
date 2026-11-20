
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.data.busi_dataset import BUSIDataset


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data/processed/ultrasound/busi_manifest.csv"

OUTPUT_DIR = (
    ROOT / "outputs/annotation_analysis/coordinate_grids"
)

TARGET_IMAGES = [
    "malignant (111).png",
    "malignant (28).png",
    "malignant (48).png",
    "normal (117).png",
    "normal (111).png",
    "benign (256).png",
]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = BUSIDataset(
        str(MANIFEST),
        split="test"
    )

    records = {
        Path(row["image_path"]).name: index
        for index, row in enumerate(dataset.records)
    }

    print("\n========== ANNOTATION COORDINATE AUDIT ==========")

    for filename in TARGET_IMAGES:

        if filename not in records:
            print("Not found in test split:", filename)
            continue

        sample = dataset[records[filename]]

        image = sample["image"].cpu().numpy()

        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * 0.5 + 0.5, 0, 1)
        image = image.mean(axis=2)

        height, width = image.shape

        fig, ax = plt.subplots(figsize=(9, 8))

        ax.imshow(
            image,
            cmap="gray",
            extent=[0, width, height, 0]
        )

        ax.set_xticks(np.arange(0, width + 1, 16))
        ax.set_yticks(np.arange(0, height + 1, 16))

        ax.grid(
            True,
            color="yellow",
            linewidth=0.45,
            alpha=0.65
        )

        ax.set_title(
            f"{filename}\n"
            "Coordinates: X horizontal | Y vertical"
        )

        ax.set_xlabel("X coordinate (pixels)")
        ax.set_ylabel("Y coordinate (pixels)")

        output_path = (
            OUTPUT_DIR
            / f"{Path(filename).stem}_grid.png"
        )

        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("Generated:", output_path)

    print("\nCoordinate audit completed.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
