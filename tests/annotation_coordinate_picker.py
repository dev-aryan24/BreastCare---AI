
from pathlib import Path

import cv2
import pandas as pd


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SELECTED_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "annotation_analysis"
    / "selected_images.csv"
)

GRID_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "annotation_analysis"
    / "coordinate_grids"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "annotation_analysis"
    / "annotation_regions.csv"
)

DISPLAY_SCALE = 3

WINDOW_NAME = "BUSI Annotation Coordinate Picker"


COLUMNS = [
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


# ==================================================
# GLOBAL MOUSE VARIABLES
# ==================================================

drawing = False
start_point = None
current_box = None

annotation_box = None
control_box = None

mode = None


# ==================================================
# DRAW RECTANGLE
# ==================================================

def draw_box(canvas, box, color):

    if box is not None:

        cv2.rectangle(
            canvas,
            (box[0], box[1]),
            (box[2], box[3]),
            color,
            1
        )


# ==================================================
# MOUSE CALLBACK
# ==================================================

def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_point
    global current_box

    if mode not in ("annotation", "control"):
        return

    x = int(x / DISPLAY_SCALE)
    y = int(y / DISPLAY_SCALE)

    x = max(0, min(223, x))
    y = max(0, min(223, y))

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True
        start_point = (x, y)
        current_box = (x, y, x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:

        current_box = (
            start_point[0],
            start_point[1],
            x,
            y
        )

    elif event == cv2.EVENT_LBUTTONUP and drawing:

        drawing = False

        if start_point is not None:

            x1 = min(start_point[0], x)
            y1 = min(start_point[1], y)

            x2 = max(start_point[0], x)
            y2 = max(start_point[1], y)

            if x2 > x1 and y2 > y1:

                current_box = (
                    x1,
                    y1,
                    x2,
                    y2
                )
            else:
                current_box = None


# ==================================================
# GRID IMAGE PATH
# ==================================================

def get_grid_path(filename):

    grid_name = (
        Path(filename).stem.replace(" ", "_")
        + "_grid.png"
    )

    return GRID_DIR / grid_name


# ==================================================
# SAVE ROW
# ==================================================

def save_row(row):

    if OUTPUT_CSV.exists():

        existing = pd.read_csv(
            OUTPUT_CSV,
            dtype=object
        )

    else:

        existing = pd.DataFrame(
            columns=COLUMNS
        )

    for column in COLUMNS:

        if column not in existing.columns:
            existing[column] = ""

    # Remove previous record for the same image.
    existing = existing[
        existing["filename"] != row["filename"]
    ]

    new_row = pd.DataFrame(
        [row],
        columns=COLUMNS
    )

    updated = pd.concat(
        [existing, new_row],
        ignore_index=True
    )

    updated = updated[COLUMNS]

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    updated.to_csv(
        OUTPUT_CSV,
        index=False
    )


# ==================================================
# SAVE ANNOTATION AND CONTROL REGIONS
# ==================================================

def save_current(filename):

    if annotation_box is None:

        print("Annotation region not selected.")
        return False

    if control_box is None:

        print("Control region not selected.")
        return False

    row = {

        "filename": filename,

        "annotation_id": 1,

        "annotation_type": "visual_annotation",

        "x1": annotation_box[0],
        "y1": annotation_box[1],
        "x2": annotation_box[2],
        "y2": annotation_box[3],

        "control_x1": control_box[0],
        "control_y1": control_box[1],
        "control_x2": control_box[2],
        "control_y2": control_box[3],

        "visual_verified": True,

        "notes": "Manually reviewed annotation and control regions."

    }

    save_row(row)

    print(f"Saved: {filename}")

    return True


# ==================================================
# SAVE IMAGE WITHOUT VISIBLE ANNOTATION
# ==================================================

def save_no_annotation(filename):

    row = {

        "filename": filename,

        "annotation_id": "",

        "annotation_type": "none",

        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",

        "control_x1": "",
        "control_y1": "",
        "control_x2": "",
        "control_y2": "",

        "visual_verified": True,

        "notes": "No visible annotation identified during manual review."

    }

    save_row(row)

    print(f"Saved as no visible annotation: {filename}")


# ==================================================
# GET PENDING IMAGES
# ==================================================

def get_pending_images():

    if not SELECTED_CSV.exists():

        raise FileNotFoundError(
            f"Selected images CSV not found: {SELECTED_CSV}"
        )

    if not GRID_DIR.exists():

        raise FileNotFoundError(
            f"Coordinate grid folder not found: {GRID_DIR}"
        )

    selected = pd.read_csv(SELECTED_CSV)

    if "filename" not in selected.columns:

        raise ValueError(
            "selected_images.csv must contain filename."
        )

    filenames = selected[
        "filename"
    ].dropna().tolist()

    if OUTPUT_CSV.exists():

        reviewed = pd.read_csv(
            OUTPUT_CSV,
            dtype=object
        )

        if not reviewed.empty:

            verified = (
                reviewed["visual_verified"]
                .astype(str)
                .str.lower()
                .isin(["true", "1"])
            )

            completed = set(
                reviewed.loc[
                    verified,
                    "filename"
                ].dropna()
            )

            filenames = [
                filename
                for filename in filenames
                if filename not in completed
            ]

    return filenames


# ==================================================
# MAIN FUNCTION
# ==================================================

def main():

    global drawing
    global start_point
    global current_box

    global annotation_box
    global control_box
    global mode

    filenames = get_pending_images()

    print("\n========================================")
    print("       BUSI ANNOTATION REVIEW")
    print("========================================")

    print("Images remaining:", len(filenames))

    print("\nCONTROLS:")
    print("A = Select annotation region")
    print("C = Select control region")
    print("N = Save and next image")
    print("S = Confirm no visible annotation")
    print("R = Reset selections")
    print("Q = Quit")
    print("========================================\n")

    if not filenames:

        print("All selected images have been reviewed.")
        return

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    current_index = 0

    while current_index < len(filenames):

        filename = filenames[current_index]

        grid_path = get_grid_path(filename)

        image = cv2.imread(
            str(grid_path)
        )

        if image is None:

            print(f"Grid image not found: {grid_path}")

            current_index += 1
            continue

        if image.shape[:2] != (224, 224):

            print(
                f"Invalid image dimensions: {filename}"
            )

            current_index += 1
            continue

        annotation_box = None
        control_box = None
        current_box = None

        mode = None
        drawing = False
        start_point = None

        print(
            f"\nReviewing {current_index + 1}/"
            f"{len(filenames)}: {filename}"
        )

        while True:

            canvas = image.copy()

            draw_box(
                canvas,
                annotation_box,
                (0, 0, 255)
            )

            draw_box(
                canvas,
                control_box,
                (0, 255, 0)
            )

            if current_box is not None:

                color = (
                    (0, 0, 255)
                    if mode == "annotation"
                    else (0, 255, 0)
                )

                draw_box(
                    canvas,
                    current_box,
                    color
                )

            cv2.putText(
                canvas,
                f"{current_index + 1}/{len(filenames)} {filename}",
                (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1
            )

            display = cv2.resize(
                canvas,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE,
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            key = cv2.waitKey(30) & 0xFF

            # --------------------------------------
            # SELECT ANNOTATION
            # --------------------------------------

            if key == ord("a"):

                mode = "annotation"
                current_box = None

                print("Draw annotation rectangle.")

            # --------------------------------------
            # SELECT CONTROL
            # --------------------------------------

            elif key == ord("c"):

                mode = "control"
                current_box = None

                print("Draw control rectangle.")

            # --------------------------------------
            # RESET
            # --------------------------------------

            elif key == ord("r"):

                annotation_box = None
                control_box = None
                current_box = None

                mode = None
                drawing = False
                start_point = None

                print("Selections reset.")

            # --------------------------------------
            # SAVE AND NEXT
            # --------------------------------------

            elif key == ord("n"):

                if save_current(filename):

                    current_index += 1
                    break

            # --------------------------------------
            # NO VISIBLE ANNOTATION
            # --------------------------------------

            elif key == ord("s"):

                save_no_annotation(filename)

                current_index += 1
                break

            # --------------------------------------
            # QUIT
            # --------------------------------------

            elif key == ord("q"):

                print("\nReview stopped.")

                cv2.destroyAllWindows()
                return

            # --------------------------------------
            # COMMIT SELECTED RECTANGLE
            # --------------------------------------

            if (
                not drawing
                and current_box is not None
                and mode == "annotation"
            ):

                annotation_box = current_box
                current_box = None

                print(
                    "Annotation region selected:",
                    annotation_box
                )

            elif (
                not drawing
                and current_box is not None
                and mode == "control"
            ):

                control_box = current_box
                current_box = None

                print(
                    "Control region selected:",
                    control_box
                )

    cv2.destroyAllWindows()

    print("\n========================================")
    print("ANNOTATION REVIEW SESSION COMPLETED")
    print("========================================")

    print("Images reviewed:", len(filenames))
    print("Output:", OUTPUT_CSV)


# ==================================================
# PROGRAM ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()
