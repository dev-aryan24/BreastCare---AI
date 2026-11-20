
import csv
import math
from pathlib import Path

import torch

from src.data.busi_dataset import BUSIDataset
from src.models.multitask_cnn import MultiTaskCNN


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data/processed/ultrasound/busi_manifest.csv"

CHECKPOINT = ROOT / "checkpoints/best_multitask_cnn_gpu.pth"

OUTPUT_DIR = ROOT / "outputs/sensitivity_analysis"

RESULTS_CSV = OUTPUT_DIR / "sensitivity_predictions.csv"
COMPARISON_CSV = OUTPUT_DIR / "sensitivity_comparison.csv"

CLASS_NAMES = ["Benign", "Malignant", "Normal"]

BORDER_FRACTION = 0.05

# Add manually verified annotation boxes here later.
# Format: "filename": [(x1, y1, x2, y2)]
# Coordinates are normalized between 0 and 1.
ANNOTATION_BOXES = {
    "malignant (111).png": [
        (33, 189, 61, 205),   # RT2 label
        (8, 76, 21, 100),     # Left marker
        (90, 46, 103, 62),    # Upper marker
        (146, 86, 162, 103),  # Right marker
        (87, 143, 108, 163),  # Lower marker and number
    ]
}


def load_model(device):

    model = MultiTaskCNN(num_classes=3).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])

    return model


def mask_border(image):

    result = image.clone()

    _, height, width = result.shape

    border_y = max(1, int(height * BORDER_FRACTION))
    border_x = max(1, int(width * BORDER_FRACTION))

    result[:, :border_y, :] = 0
    result[:, -border_y:, :] = 0
    result[:, :, :border_x] = 0
    result[:, :, -border_x:] = 0

    return result


def mask_center_control(image):

    result = image.clone()

    _, height, width = result.shape

    border_x = max(1, int(width * BORDER_FRACTION))
    border_y = max(1, int(height * BORDER_FRACTION))

    border_area = (
        height * width
        - (height - 2 * border_y) * (width - 2 * border_x)
    )

    side = int(math.sqrt(border_area))

    center_x = width // 2
    center_y = height // 2

    x1 = max(0, center_x - side // 2)
    x2 = min(width, x1 + side)

    y1 = max(0, center_y - side // 2)
    y2 = min(height, y1 + side)

    result[:, y1:y2, x1:x2] = 0

    return result


def mask_annotations(image, filename):

    result = image.clone()

    _, height, width = result.shape

    boxes = ANNOTATION_BOXES.get(filename, [])

    for x1, y1, x2, y2 in boxes:

        left = max(0, min(width, x1))
        right = max(0, min(width, x2))

        top = max(0, min(height, y1))
        bottom = max(0, min(height, y2))

        if right > left and bottom > top:
            result[:, top:bottom, left:right] = 0

    return result, len(boxes)


def evaluate_image(model, image, device):

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        class_logits, segmentation_logits = model(image)

        probabilities = torch.softmax(class_logits, dim=1)

        confidence, prediction = probabilities.max(dim=1)

        segmentation = torch.sigmoid(segmentation_logits)

        predicted_pixels = int(
            (segmentation >= 0.5).sum().item()
        )

    return {
        "logits": class_logits[0].cpu().tolist(),
        "prediction": int(prediction.item()),
        "confidence": float(confidence.item()),
        "predicted_pixels": predicted_pixels
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n========== SENSITIVITY ANALYSIS ==========")
    print("Device:", device)

    dataset = BUSIDataset(
        str(MANIFEST),
        split="test"
    )

    model = load_model(device)

    all_results = []
    comparison_results = []

    for index in range(len(dataset)):

        sample = dataset[index]

        image = sample["image"]
        filename = Path(sample["image_path"]).name
        actual_class = int(sample["label"].item())

        original = evaluate_image(model, image, device)

        border_image = mask_border(image)

        border_result = evaluate_image(
            model, border_image, device
        )

        control_image = mask_center_control(image)

        control_result = evaluate_image(
            model, control_image, device
        )

        conditions = {
            "original": original,
            "border_masked": border_result,
            "center_control_masked": control_result
        }

        annotation_count = 0

        if filename in ANNOTATION_BOXES:

            annotation_image, annotation_count = mask_annotations(
                image, filename
            )

            conditions["annotation_masked"] = evaluate_image(
                model, annotation_image, device
            )

        for condition, result in conditions.items():

            all_results.append({
                "filename": filename,
                "actual_class": CLASS_NAMES[actual_class],
                "experiment": condition,
                "predicted_class": CLASS_NAMES[result["prediction"]],
                "confidence": round(result["confidence"], 6),
                "classification_correct": (
                    actual_class == result["prediction"]
                ),
                "predicted_mask_pixels": result["predicted_pixels"],
                "logit_benign": round(result["logits"][0], 6),
                "logit_malignant": round(result["logits"][1], 6),
                "logit_normal": round(result["logits"][2], 6)
            })

        comparison_results.append({
            "filename": filename,
            "actual_class": CLASS_NAMES[actual_class],

            "original_prediction": CLASS_NAMES[
                original["prediction"]
            ],

            "border_prediction": CLASS_NAMES[
                border_result["prediction"]
            ],

            "control_prediction": CLASS_NAMES[
                control_result["prediction"]
            ],

            "original_confidence": round(
                original["confidence"], 6
            ),

            "border_confidence": round(
                border_result["confidence"], 6
            ),

            "control_confidence": round(
                control_result["confidence"], 6
            ),

            "border_confidence_change": round(
                border_result["confidence"] -
                original["confidence"], 6
            ),

            "control_confidence_change": round(
                control_result["confidence"] -
                original["confidence"], 6
            ),

            "border_prediction_changed": (
                original["prediction"] !=
                border_result["prediction"]
            ),

            "control_prediction_changed": (
                original["prediction"] !=
                control_result["prediction"]
            ),

            "original_mask_pixels": original["predicted_pixels"],

            "border_mask_pixels": border_result["predicted_pixels"],

            "control_mask_pixels": control_result["predicted_pixels"],

            "annotation_boxes_configured": annotation_count
        })

        if (index + 1) % 20 == 0:
            print(f"Processed {index + 1}/{len(dataset)} images")

    def write_csv(path, rows):

        if not rows:
            return

        with path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(rows)

    write_csv(RESULTS_CSV, all_results)
    write_csv(COMPARISON_CSV, comparison_results)

    print("\n========== RESULTS ==========")
    print("Test images:", len(dataset))
    print("Prediction records:", len(all_results))
    print("Comparison records:", len(comparison_results))

    for condition in [
        "original",
        "border_masked",
        "center_control_masked"
    ]:

        records = [
            row for row in all_results
            if row["experiment"] == condition
        ]

        correct = sum(
            row["classification_correct"]
            for row in records
        )

        accuracy = correct / len(records) if records else 0

        print(
            f"{condition}: "
            f"{correct}/{len(records)} "
            f"({accuracy:.4f})"
        )

    changed_border = sum(
        row["border_prediction_changed"]
        for row in comparison_results
    )

    changed_control = sum(
        row["control_prediction_changed"]
        for row in comparison_results
    )

    print("\nBorder prediction changes:", changed_border)
    print("Control prediction changes:", changed_control)

    print("\nPrediction CSV:", RESULTS_CSV)
    print("Comparison CSV:", COMPARISON_CSV)

    print("\n========== EXPERIMENT COMPLETED ==========")


if __name__ == "__main__":
    main()
