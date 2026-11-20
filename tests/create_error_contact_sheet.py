
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "outputs/test_error_analysis/images"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs/test_error_analysis/error_contact_sheet.png"
)

TARGETS = [
    "malignant (111)",
    "malignant (28)",
    "malignant (48)",
    "normal (117)",
    "normal (111)",
    "benign (256)"
]

TILE_WIDTH = 900
TILE_HEIGHT = 260
COLUMNS = 2


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Overlay directory not found: {IMAGE_DIR}"
        )

    files = list(IMAGE_DIR.glob("*_analysis.png"))

    selected = []

    for target in TARGETS:
        matches = [
            path for path in files
            if target.lower() in path.name.lower()
        ]

        if matches:
            selected.append((target, matches[0]))
        else:
            print("Missing overlay:", target)

    if not selected:
        raise RuntimeError("No matching overlay images found.")

    rows = (len(selected) + COLUMNS - 1) // COLUMNS

    sheet = Image.new(
        "RGB",
        (COLUMNS * TILE_WIDTH, rows * TILE_HEIGHT),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for index, (target, path) in enumerate(selected):
        image = Image.open(path).convert("RGB")

        image.thumbnail(
            (TILE_WIDTH - 20, TILE_HEIGHT - 55)
        )

        x = (index % COLUMNS) * TILE_WIDTH
        y = (index // COLUMNS) * TILE_HEIGHT

        draw.text(
            (x + 10, y + 5),
            target,
            fill="black",
            font=font
        )

        image_x = x + (TILE_WIDTH - image.width) // 2
        image_y = y + 40

        sheet.paste(image, (image_x, image_y))

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sheet.save(OUTPUT_PATH)

    print("\nContact sheet created successfully.")
    print("Images included:", len(selected))
    print("Saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
