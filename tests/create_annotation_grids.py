
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "outputs/annotation_analysis/selected_images.csv"

OUTPUT_DIR = (
    ROOT / "outputs/annotation_analysis/coordinate_grids"
)

IMAGE_SIZE = 224
GRID_STEP = 16

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_grid(image_path, output_path, metadata):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    overlay = image.convert("RGBA")

    grid_layer = Image.new(
        "RGBA",
        overlay.size,
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(grid_layer)

    try:
        font = ImageFont.truetype("arial.ttf", 9)
    except OSError:
        font = ImageFont.load_default()

    # Draw vertical and horizontal coordinate lines
    for coordinate in range(0, IMAGE_SIZE + 1, GRID_STEP):

        draw.line(
            [(coordinate, 0), (coordinate, IMAGE_SIZE)],
            fill=(255, 255, 0, 110),
            width=1
        )

        draw.line(
            [(0, coordinate), (IMAGE_SIZE, coordinate)],
            fill=(255, 255, 0, 110),
            width=1
        )

        if coordinate < IMAGE_SIZE:
            draw.text(
                (coordinate + 1, 1),
                str(coordinate),
                fill=(255, 255, 255, 255),
                font=font
            )

            draw.text(
                (1, coordinate + 1),
                str(coordinate),
                fill=(255, 255, 255, 255),
                font=font
            )

    result = Image.alpha_composite(
        overlay,
        grid_layer
    ).convert("RGB")

    result.save(output_path)


def main():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    for _, record in df.iterrows():

        filename = record["filename"]
        image_path = Path(record["image_path"])

        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue

        safe_name = Path(filename).stem.replace(" ", "_")

        output_path = (
            OUTPUT_DIR / f"{safe_name}_grid.png"
        )

        create_grid(
            image_path,
            output_path,
            record
        )

        print(f"Created: {output_path.name}")

    print("\n========== COORDINATE GRIDS COMPLETED ==========")
    print(f"Images processed: {len(df)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

