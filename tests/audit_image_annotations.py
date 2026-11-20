
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BUSI"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "gradcam_analysis"
    / "original_image_audit.png"
)

TARGET_IMAGES = [
    "malignant (111).png",
    "malignant (28).png",
    "malignant (48).png",
    "normal (117).png",
    "normal (111).png",
    "benign (256).png"
]

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 300
LABEL_HEIGHT = 45
PADDING = 20
COLUMNS = 2


def load_font():

    try:
        return ImageFont.truetype(
            "arial.ttf",
            20
        )

    except OSError:
        return ImageFont.load_default()


def find_dataset_images():

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"BUSI dataset directory not found: {DATASET_ROOT}"
        )

    image_files = {}

    for path in DATASET_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg"
        }:
            continue

        image_files.setdefault(
            path.name,
            []
        ).append(path)

    return image_files


def create_contact_sheet(image_files):

    rows = (
        len(TARGET_IMAGES) + COLUMNS - 1
    ) // COLUMNS

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

    for index, filename in enumerate(TARGET_IMAGES):

        matching_paths = image_files.get(
            filename,
            []
        )

        if not matching_paths:

            print(f"Image not found: {filename}")
            continue

        if len(matching_paths) > 1:

            print(
                f"Multiple files found for {filename}:"
            )

            for path in matching_paths:
                print(" ", path)

            print("Skipping ambiguous filename.")
            continue

        image_path = matching_paths[0]

        try:

            with Image.open(image_path) as source:
                image = source.convert("RGB")

            image.thumbnail(
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                Image.Resampling.LANCZOS
            )

        except Exception as error:

            print(
                f"Unable to read {filename}: {error}"
            )

            continue

        column = index % COLUMNS
        row = index // COLUMNS

        x = (
            PADDING
            + column * (IMAGE_WIDTH + PADDING)
        )

        y = (
            PADDING
            + row * (IMAGE_HEIGHT + LABEL_HEIGHT)
        )

        draw.text(
            (x, y),
            filename,
            fill="black",
            font=font
        )

        image_x = (
            x + (IMAGE_WIDTH - image.width) // 2
        )

        image_y = y + LABEL_HEIGHT

        sheet.paste(
            image,
            (image_x, image_y)
        )

        print("Processed:", filename)
        print("Source:", image_path)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sheet.save(
        OUTPUT_PATH
    )

    print("\n========== IMAGE AUDIT COMPLETED ==========")
    print("Dataset:", DATASET_ROOT)
    print("Output:", OUTPUT_PATH)


def main():

    print("Scanning BUSI dataset...")

    image_files = find_dataset_images()

    print(
        "Total unique image filenames:",
        len(image_files)
    )

    create_contact_sheet(image_files)


if __name__ == "__main__":
    main()
