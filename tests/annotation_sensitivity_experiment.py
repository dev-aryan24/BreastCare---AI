
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src.data.busi_dataset import BUSIDataset
from src.models.multitask_cnn import MultiTaskCNN


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data/processed/ultrasound/busi_manifest.csv"
CHECKPOINT = ROOT / "checkpoints/best_multitask_cnn_gpu.pth"

REGION_CSV = ROOT / "outputs/annotation_analysis/annotation_regions.csv"

OUTPUT_DIR = ROOT / "outputs/annotation_analysis"
IMAGE_DIR = OUTPUT_DIR / "visualizations"

TARGET_IMAGE = "malignant (111).png"

CLASS_NAMES = ["Benign", "Malignant", "Normal"]


def load_validated_regions():

    with REGION_CSV.open(
        "r", newline="", encoding="utf-8"
    ) as file:

        records = list(csv.DictReader(file))

    matching = [
        row for row in records
        if row["filename"] == TARGET_IMAGE
    ]

    if len(matching) != 1:
        raise ValueError(
            f"Expected exactly one coordinate record; found {len(matching)}."
        )

    row = matching[0]

    if str(row["visual_verified"]).strip().lower() not in (
        "true", "1", "yes"
    ):
        raise ValueError("Annotation region has not been visually verified.")

    if row["annotation_type"] != "visual_annotation":
        raise ValueError("Record is not marked as visual_annotation.")

    annotation_box = tuple(
        int(float(row[key]))
        for key in ["x1", "y1", "x2", "y2"]
    )

    control_box = tuple(
        int(float(row[key]))
        for key in [
            "control_x1", "control_y1",
            "control_x2", "control_y2"
        ]
    )

    return annotation_box, control_box


def validate_regions(annotation_box, control_box, target_mask):

    height, width = target_mask.shape[-2:]

    for name, box in [
        ("Annotation", annotation_box),
        ("Control", control_box)
    ]:

        x1, y1, x2, y2 = box

        if not (0 <= x1 < x2 <= width):
            raise ValueError(f"{name} X coordinates are invalid.")

        if not (0 <= y1 < y2 <= height):
            raise ValueError(f"{name} Y coordinates are invalid.")

    ax1, ay1, ax2, ay2 = annotation_box
    cx1, cy1, cx2, cy2 = control_box

    rectangles_overlap = (
        ax1 < cx2 and cx1 < ax2
        and ay1 < cy2 and cy1 < ay2
    )

    if rectangles_overlap:
        raise ValueError("Annotation and control rectangles overlap.")

    lesion = target_mask.squeeze().cpu().numpy() > 0.5

    for name, box in [
        ("Annotation", annotation_box),
        ("Control", control_box)
    ]:

        x1, y1, x2, y2 = box

        region = lesion[y1:y2, x1:x2]

        overlap = int(region.sum())

        if overlap != 0:
            raise ValueError(
                f"{name} region overlaps {overlap} lesion pixels."
            )

        print(f"{name} lesion overlap: {overlap} pixels")

    print("Geometric validation: PASSED")


def apply_blur(image, boxes):

    result = image.clone()

    blurred = F.avg_pool2d(
        image.unsqueeze(0),
        kernel_size=9,
        stride=1,
        padding=4
    )[0]

    for x1, y1, x2, y2 in boxes:
        result[:, y1:y2, x1:x2] = (
            blurred[:, y1:y2, x1:x2]
        )

    return result


def apply_constant_mask(image, boxes):

    result = image.clone()

    for x1, y1, x2, y2 in boxes:
        result[:, y1:y2, x1:x2] = 0

    return result


def evaluate(model, image, target_mask, device):

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        logits, segmentation_logits = model(image)

        probabilities = torch.softmax(logits, dim=1)

        confidence, prediction = probabilities.max(dim=1)

        predicted_mask = (
            torch.sigmoid(segmentation_logits) >= 0.5
        ).float()

        target_mask = target_mask.unsqueeze(0).to(device)

        intersection = (
            predicted_mask * target_mask
        ).sum()

        denominator = (
            predicted_mask.sum() + target_mask.sum()
        )

        dice = (
            (2 * intersection + 1e-7)
            / (denominator + 1e-7)
        )

        return {
            "prediction": int(prediction.item()),
            "confidence": float(confidence.item()),
            "probabilities": probabilities[0].cpu().tolist(),
            "predicted_pixels": int(predicted_mask.sum().item()),
            "dice": float(dice.item())
        }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    annotation_box, control_box = load_validated_regions()

    dataset = BUSIDataset(
        str(MANIFEST),
        split="test"
    )

    matching = [
        i for i, row in enumerate(dataset.records)
        if Path(row["image_path"]).name == TARGET_IMAGE
    ]

    if len(matching) != 1:
        raise ValueError(
            f"Expected one test image; found {len(matching)}."
        )

    sample = dataset[matching[0]]

    image = sample["image"]
    mask = sample["mask"]

    actual_class = int(sample["label"].item())

    validate_regions(
        annotation_box,
        control_box,
        mask
    )

    model = MultiTaskCNN(num_classes=3).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    annotation_boxes = [annotation_box]
    control_boxes = [control_box]

    conditions = {
        "original": image,

        "annotation_blurred": apply_blur(
            image, annotation_boxes
        ),

        "control_blurred": apply_blur(
            image, control_boxes
        ),

        "annotation_constant": apply_constant_mask(
            image, annotation_boxes
        )
    }

    results = []
    visual_images = []

    print("\n========== VALIDATED ANNOTATION SENSITIVITY ==========")
    print("Image:", TARGET_IMAGE)
    print("Actual class:", CLASS_NAMES[actual_class])
    print("Annotation coordinates:", annotation_box)
    print("Control coordinates:", control_box)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Device:", device)

    for name, modified_image in conditions.items():

        result = evaluate(
            model,
            modified_image,
            mask,
            device
        )

        results.append({
            "image": TARGET_IMAGE,
            "experiment": name,
            "actual_class": CLASS_NAMES[actual_class],
            "predicted_class": CLASS_NAMES[result["prediction"]],
            "confidence": round(result["confidence"], 6),
            "benign_probability": round(result["probabilities"][0], 6),
            "malignant_probability": round(result["probabilities"][1], 6),
            "normal_probability": round(result["probabilities"][2], 6),
            "predicted_mask_pixels": result["predicted_pixels"],
            "segmentation_dice": round(result["dice"], 6)
        })

        display = modified_image.cpu().numpy()
        display = np.transpose(display, (1, 2, 0))
        display = np.clip(display * 0.5 + 0.5, 0, 1)
        display = display.mean(axis=2)

        visual_images.append((name, display, result))

        print(f"\nExperiment: {name}")
        print("Prediction:", CLASS_NAMES[result["prediction"]])
        print("Confidence:", round(result["confidence"], 4))
        print("Mask pixels:", result["predicted_pixels"])
        print("Segmentation Dice:", round(result["dice"], 4))

    csv_path = (
        OUTPUT_DIR /
        "annotation_sensitivity_results_validated.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    fig, axes = plt.subplots(
        1, len(visual_images),
        figsize=(19, 5)
    )

    for ax, (name, display, result) in zip(
        axes, visual_images
    ):

        ax.imshow(display, cmap="gray")

        ax.set_title(
            f"{name}\n"
            f"{CLASS_NAMES[result['prediction']]}\n"
            f"Confidence: {result['confidence']:.3f}"
        )

        ax.axis("off")

    fig.tight_layout()

    visualization_path = (
        IMAGE_DIR /
        "annotation_comparison_validated.png"
    )

    fig.savefig(
        visualization_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\n========== EXPERIMENT COMPLETED ==========")
    print("CSV:", csv_path)
    print("Visualization:", visualization_path)


if __name__ == "__main__":
    main()
