
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "outputs/annotation_analysis/selected_images.csv"
OUTPUT_DIR = ROOT / "outputs/annotation_analysis"
OUTPUT = OUTPUT_DIR / "annotation_audit.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

audit = pd.DataFrame({
    "filename": df["filename"],
    "actual_class": df["actual_class"],
    "predicted_class": df["predicted_class"],
    "selection_group": df["selection_group"],

    "annotation_type": "",
    "annotation_x1": "",
    "annotation_y1": "",
    "annotation_x2": "",
    "annotation_y2": "",

    "control_x1": "",
    "control_y1": "",
    "control_x2": "",
    "control_y2": "",

    "lesion_overlap_pixels": "",
    "annotation_verified": "Pending",
    "review_notes": ""
})

audit.to_csv(OUTPUT, index=False)

print("========== ANNOTATION AUDIT TEMPLATE ==========")
print(f"Images: {len(audit)}")
print(f"Output: {OUTPUT}")
print("\nAnnotation audit template created successfully.")
