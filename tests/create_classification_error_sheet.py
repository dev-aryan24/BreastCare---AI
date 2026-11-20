
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "outputs/test_error_analysis/test_predictions.csv"
)

OUTPUT_DIR = ROOT / "outputs/test_error_analysis"

OUTPUT_IMAGE = OUTPUT_DIR / "classification_error_contact_sheet.png"

OUTPUT_CSV = OUTPUT_DIR / "classification_errors_only.csv"

IMAGE_WIDTH = 360
IMAGE_HEIGHT = 260
LABEL_HEIGHT = 75
PADDING = 18
COLUMNS = 3


def load_font(size=17):

    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def load_errors():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Prediction CSV not found: {CSV_PATH}"
        )

    errors = []

    with CSV_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "filename",
            "image_path",
            "actual_class",
            "predicted_class",
            "confidence",
            "classification_correct",
            "segmentation_dice"
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        for row in reader:

            if row["classification_correct"].strip().lower() == "false":
                errors.append(row)

    return errors


def save_error_csv(errors):

    if not errors:
        print("No classification errors found.")
        return

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=errors[0].keys()
        )

        writer.writeheader()
        writer.writerows(errors)


def create_contact_sheet(errors):

    if not errors:
        print("No images available for contact sheet.")
        return

    rows = (len(errors) + COLUMNS - 1) // COLUMNS

    sheet_width = (
        COLUMNS * IMAGE_WIDTH
        + (COLUMNS + 1) * PADDING
    )

    sheet_height = (
        rows * (IMAGE_HEIGHT + LABEL_HEIGHT)
        + (rows + 1) * PADDING
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)
    font = load_font()
    small_font = load_font(14)

    for index, record in enumerate(errors):

        image_path = Path(record["image_path"])

        if not image_path.exists():
            print("Image missing:", image_path)
            continue

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")

            image.thumbnail(
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                Image.Resampling.LANCZOS
            )

        except Exception as error:
            print("Unable to read:", image_path, error)
            continue

        column = index % COLUMNS
        row = index // COLUMNS

        x = PADDING + column * (IMAGE_WIDTH + PADDING)
        y = PADDING + row * (IMAGE_HEIGHT + LABEL_HEIGHT)

        confidence = float(record["confidence"])
        dice = float(record["segmentation_dice"])

        draw.text(
            (x, y),
            record["filename"],
            fill="black",
            font=font
        )

        draw.text(
            (x, y + 25),
            f"Actual: {record['actual_class']} | "
            f"Predicted: {record['predicted_class']}",
            fill="red",
            font=small_font
        )

        draw.text(
            (x, y + 45),
            f"Confidence: {confidence:.4f} | "
            f"Dice: {dice:.4f}",
            fill="black",
            font=small_font
        )

        image_x = x + (IMAGE_WIDTH - image.width) // 2
        image_y = y + LABEL_HEIGHT

        sheet.paste(image, (image_x, image_y))

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    sheet.save(OUTPUT_IMAGE)

    print("Contact sheet saved:", OUTPUT_IMAGE)


def print_error_summary(errors):

    print("\n========== CLASSIFICATION ERROR SUMMARY ==========")

    print("Total classification errors:", len(errors))

    error_pairs = {}

    for record in errors:

        pair = (
            record["actual_class"],
            record["predicted_class"]
        )

        error_pairs[pair] = error_pairs.get(pair, 0) + 1

    print("\nMisclassification breakdown:")

    for (actual, predicted), count in sorted(error_pairs.items()):

        print(
            f"{actual} -> {predicted}: {count}"
        )


def main():

    print("Loading test predictions...")

    errors = load_errors()

    print_error_summary(errors)

    save_error_csv(errors)

    create_contact_sheet(errors)

    print("\n========== AUDIT COMPLETED ==========")


if __name__ == "__main__":
    main()
