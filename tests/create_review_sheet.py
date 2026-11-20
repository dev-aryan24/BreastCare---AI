
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "outputs/annotation_analysis/coordinate_grids"
OUTPUT = ROOT / "outputs/annotation_analysis/first_four_review.png"

files = [
    "malignant_(111)_grid.png",
    "malignant_(2)_grid.png",
    "benign_(310)_grid.png",
    "normal_(117)_grid.png"
]

sheet = Image.new("RGB", (960, 1020), "white")
draw = ImageDraw.Draw(sheet)

positions = [
    (0, 0),
    (480, 0),
    (0, 510),
    (480, 510)
]

for filename, (x, y) in zip(files, positions):

    path = SOURCE / filename

    if not path.exists():
        raise FileNotFoundError(path)

    image = Image.open(path).convert("RGB")
    image = image.resize((448, 448))

    sheet.paste(image, (x + 16, y + 35))
    draw.text(
        (x + 16, y + 10),
        filename,
        fill="black"
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT)

print(f"Review sheet saved: {OUTPUT}")
