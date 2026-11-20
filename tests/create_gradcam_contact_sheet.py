
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "outputs/gradcam_analysis/images"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs/gradcam_analysis/gradcam_contact_sheet.png"
)

TARGET_IMAGES = [
    "malignant (111)_gradcam.png",
    "malignant (28)_gradcam.png",
    "malignant (48)_gradcam.png",
    "normal (117)_gradcam.png",
    "normal (111)_gradcam.png",
    "benign (256)_gradcam.png"
]

THUMB_WIDTH = 900
THUMB_HEIGHT = 300
LABEL_HEIGHT = 45
PADDING = 20
COLUMNS = 2


def main():

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    rows = (len(TARGET_IMAGES) + COLUMNS - 1) // COLUMNS

    sheet_width = (
        COLUMNS * THUMB_WIDTH
        + (COLUMNS + 1) * PADDING
    )

    sheet_height = (
        rows * (THUMB_HEIGHT + LABEL_HEIGHT)
        + (rows + 1) * PADDING
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, filename in enumerate(TARGET_IMAGES):

        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            print("Missing:", image_path)
            continue

        image = Image.open(image_path).convert("RGB")

        image.thumbnail(
            (THUMB_WIDTH, THUMB_HEIGHT),
            Image.Resampling.LANCZOS
        )

        column = index % COLUMNS
        row = index // COLUMNS

        x = PADDING + column * (
            THUMB_WIDTH + PADDING
        )

        y = PADDING + row * (
            THUMB_HEIGHT + LABEL_HEIGHT
        )

        draw.text(
            (x, y),
            filename,
            fill="black",
            font=font
        )

        image_x = x + (THUMB_WIDTH - image.width) // 2
        image_y = y + LABEL_HEIGHT

        sheet.paste(
            image,
            (image_x, image_y)
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sheet.save(OUTPUT_PATH)

    print("Contact sheet saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
