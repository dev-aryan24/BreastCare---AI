
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "outputs/annotation_analysis/selected_images.csv"
OUTPUT = ROOT / "outputs/annotation_analysis/annotation_regions.csv"

df = pd.read_csv(INPUT)

columns = [
    "filename",
    "annotation_id",
    "annotation_type",
    "x1",
    "y1",
    "x2",
    "y2",
    "control_x1",
    "control_y1",
    "control_x2",
    "control_y2",
    "visual_verified",
    "notes"
]

rows = []

for _, record in df.iterrows():
    rows.append({
        "filename": record["filename"],
        "annotation_id": 1,
        "annotation_type": "",
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "control_x1": "",
        "control_y1": "",
        "control_x2": "",
        "control_y2": "",
        "visual_verified": "Pending",
        "notes": ""
    })

audit = pd.DataFrame(rows, columns=columns)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUTPUT, index=False)

print("ANNOTATION REGION TEMPLATE CREATED")
print("Images:", len(audit))
print("Output:", OUTPUT)
