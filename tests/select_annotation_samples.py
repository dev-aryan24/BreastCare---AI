
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "outputs/test_error_analysis/test_predictions.csv"

OUTPUT_DIR = ROOT / "outputs/annotation_analysis"
OUTPUT_CSV = OUTPUT_DIR / "selected_images.csv"
CONTACT_SHEET = OUTPUT_DIR / "selected_images_contact_sheet.png"

RANDOM_SEED = 42
SAMPLES_PER_GROUP = 4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_group(data, group_name, count):
    if len(data) < count:
        raise ValueError(
            f"{group_name}: requested {count}, "
            f"but only {len(data)} records are available."
        )

    selected = data.sample(
        n=count,
        random_state=RANDOM_SEED
    ).copy()

    selected["selection_group"] = group_name
    return selected


def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    required_columns = [
        "filename",
        "image_path",
        "actual_class",
        "predicted_class",
        "confidence",
        "classification_correct",
        "segmentation_dice"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df["classification_correct"] = (
        df["classification_correct"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    malignant = df[df["actual_class"] == "Malignant"]
    benign = df[df["actual_class"] == "Benign"]
    normal = df[df["actual_class"] == "Normal"]

    selected_groups = []

    selected_groups.append(
        sample_group(
            malignant[~malignant["classification_correct"]],
            "Misclassified Malignant",
            SAMPLES_PER_GROUP
        )
    )

    selected_groups.append(
        sample_group(
            malignant[malignant["classification_correct"]],
            "Correct Malignant",
            SAMPLES_PER_GROUP
        )
    )

    selected_groups.append(
        sample_group(
            benign[~benign["classification_correct"]],
            "Misclassified Benign",
            SAMPLES_PER_GROUP
        )
    )

    selected_groups.append(
        sample_group(
            benign[benign["classification_correct"]],
            "Correct Benign",
            SAMPLES_PER_GROUP
        )
    )

    normal_wrong = normal[~normal["classification_correct"]]
    normal_correct = normal[normal["classification_correct"]]

    normal_selected = pd.concat([
        sample_group(normal_wrong, "Normal - Misclassified", 2),
        sample_group(normal_correct, "Normal - Correct", 2)
    ])

    selected_groups.append(normal_selected)

    selected = pd.concat(
        selected_groups,
        ignore_index=True
    )

    selected.to_csv(OUTPUT_CSV, index=False)

    # Create contact sheet
    columns = 4
    rows = len(selected) // columns

    tile_width = 320
    tile_height = 280

    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    for index, record in selected.iterrows():

        image_path = Path(record["image_path"])

        if not image_path.exists():
            print(f"WARNING: Image not found: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        image.thumbnail((290, 215))

        column = index % columns
        row = index // columns

        x = column * tile_width
        y = row * tile_height

        image_x = x + (tile_width - image.width) // 2

        sheet.paste(image, (image_x, y + 5))

        text = [
            record["filename"],
            record["selection_group"],
            f"Actual: {record['actual_class']}",
            f"Predicted: {record['predicted_class']}",
            f"Confidence: {record['confidence']:.4f}"
        ]

        text_y = y + 225

        for line in text:
            draw.text(
                (x + 8, text_y),
                str(line),
                fill="black",
                font=font
            )
            text_y += 17

    sheet.save(CONTACT_SHEET)

    print("\n========== IMAGE SELECTION ==========")
    print(f"Total test records: {len(df)}")
    print(f"Selected images: {len(selected)}")

    print("\nGroup distribution:")
    print(selected["selection_group"].value_counts().to_string())

    print("\nSelected images:")
    print(
        selected[
            ["filename", "selection_group", "actual_class",
             "predicted_class", "confidence"]
        ].to_string(index=False)
    )

    print("\n========== OUTPUT FILES ==========")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Contact sheet: {CONTACT_SHEET}")

    print("\nIMAGE SELECTION COMPLETED")


if __name__ == "__main__":
    main()
