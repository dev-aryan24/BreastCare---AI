
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/raw/BUSI"
INPUT_CSV = ROOT / "outputs/annotation_analysis/selected_images.csv"

OUTPUT_DIR = ROOT / "outputs/annotation_analysis"
OUTPUT_CSV = OUTPUT_DIR / "mask_inventory.csv"

IMAGE_SIZE = (224, 224)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)
results = []

for _, record in df.iterrows():

    filename = record["filename"]
    class_name = record["actual_class"].lower()

    image_path = DATA_DIR / class_name / filename

    if not image_path.exists():
        results.append({
            "filename": filename,
            "mask_count": 0,
            "mask_files": "",
            "mask_status": "IMAGE_MISSING",
            "foreground_pixels": 0
        })
        continue

    stem = Path(filename).stem
    mask_paths = sorted(
        (DATA_DIR / class_name).glob(f"{stem}_mask*.png")
    )

    combined = np.zeros(
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        dtype=np.uint8
    )

    original_dimensions = []
    readable_count = 0

    for mask_path in mask_paths:

        try:
            mask = Image.open(mask_path).convert("L")
            original_dimensions.append(str(mask.size))

            mask = mask.resize(
                IMAGE_SIZE,
                Image.Resampling.NEAREST
            )

            mask_array = np.asarray(mask)
            binary_mask = (mask_array > 0).astype(np.uint8)

            combined = np.maximum(combined, binary_mask)
            readable_count += 1

        except Exception as error:
            print(f"ERROR: {mask_path.name}: {error}")

    foreground_pixels = int(combined.sum())

    if not mask_paths:
        status = "MASK_MISSING"
    elif readable_count != len(mask_paths):
        status = "MASK_READ_ERROR"
    else:
        status = "OK"

    results.append({
        "filename": filename,
        "mask_count": len(mask_paths),
        "mask_files": "; ".join(
            path.name for path in mask_paths
        ),
        "original_mask_dimensions": "; ".join(
            original_dimensions
        ),
        "foreground_pixels": foreground_pixels,
        "mask_status": status
    })

    print(
        f"{filename}: "
        f"{len(mask_paths)} masks, "
        f"{foreground_pixels} foreground pixels, "
        f"{status}"
    )

inventory = pd.DataFrame(results)
inventory.to_csv(OUTPUT_CSV, index=False)

print("\n========== MASK INVENTORY ==========")
print(f"Images checked: {len(inventory)}")
print(f"Valid: {(inventory['mask_status'] == 'OK').sum()}")
print(f"Missing: {(inventory['mask_status'] == 'MASK_MISSING').sum()}")
print(f"Errors: {(inventory['mask_status'] == 'MASK_READ_ERROR').sum()}")
print(f"Output: {OUTPUT_CSV}")
