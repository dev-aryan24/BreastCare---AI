
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CSV = ROOT / "outputs/annotation_analysis/annotation_regions.csv"
SELECTED = ROOT / "outputs/annotation_analysis/selected_images.csv"
OUTPUT = ROOT / "outputs/annotation_analysis/coordinate_validation_visual.png"

filename = "malignant (111).png"

annotations = pd.read_csv(CSV)
selected = pd.read_csv(SELECTED)

row = annotations[annotations.filename == filename].iloc[0]
image_row = selected[selected.filename == filename].iloc[0]

image_path = Path(image_row["image_path"])
if not image_path.is_absolute():
    image_path = ROOT / image_path

image = cv2.imread(str(image_path))
image = cv2.resize(image, (224, 224))

mask_paths = list(
    image_path.parent.glob(image_path.stem + "_mask*.png")
)

mask = np.zeros((224, 224), dtype=np.uint8)

for path in mask_paths:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (224, 224), interpolation=cv2.INTER_NEAREST)
    mask = np.maximum(mask, (m > 0).astype(np.uint8))

overlay = image.copy()
overlay[mask > 0] = (
    0.5 * overlay[mask > 0] + 0.5 * np.array([0, 255, 0])
).astype(np.uint8)

ax1, ay1, ax2, ay2 = [
    int(row[k]) for k in ["x1", "y1", "x2", "y2"]
]

cx1, cy1, cx2, cy2 = [
    int(row[k]) for k in [
        "control_x1", "control_y1",
        "control_x2", "control_y2"
    ]
]

cv2.rectangle(
    overlay, (ax1, ay1), (ax2, ay2), (0, 0, 255), 1
)

cv2.rectangle(
    overlay, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1
)

cv2.putText(
    overlay, "Red: Annotation | Blue: Control | Green: Lesion",
    (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.27,
    (255, 255, 255), 1
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(OUTPUT), overlay)

print("Saved:", OUTPUT)
print("Image:", image.shape)
print("Mask pixels:", int(mask.sum()))

