
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.busi_dataset import create_dataloaders
from src.models.multitask_cnn import MultiTaskCNN


MANIFEST = (
    PROJECT_ROOT
    / "data/processed/ultrasound/busi_manifest.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/best_multitask_cnn_gpu.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/threshold_analysis"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "epoch12_threshold_sweep.csv"
)

CLASS_NAMES = ["Benign", "Malignant", "Normal"]

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _, val_loader, _ = create_dataloaders(
        str(MANIFEST),
        batch_size=2
    )

    model = MultiTaskCNN(num_classes=3).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []

    total_dice = {t: 0.0 for t in THRESHOLDS}
    lesion_count = 0

    normal_fp = {t: 0 for t in THRESHOLDS}
    normal_predicted_pixels = {t: 0 for t in THRESHOLDS}
    normal_count = 0

    classification_confusion = torch.zeros(
        3, 3, dtype=torch.int64
    )

    with torch.inference_mode():

        for batch in val_loader:

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            class_logits, segmentation_logits = model(images)

            predictions = class_logits.argmax(dim=1)

            for actual, predicted in zip(
                labels.cpu(),
                predictions.cpu()
            ):
                classification_confusion[
                    actual, predicted
                ] += 1

            probabilities = torch.sigmoid(
                segmentation_logits
            )

            target_flat = masks.flatten(1)
            target_area = target_flat.sum(dim=1)

            lesion_images = (
                (labels != 2) & (target_area > 0)
            )

            normal_images = labels == 2

            lesion_count += lesion_images.sum().item()
            normal_count += normal_images.sum().item()

            for threshold in THRESHOLDS:

                predicted_masks = (
                    probabilities >= threshold
                ).float()

                predicted_flat = predicted_masks.flatten(1)

                intersection = (
                    predicted_flat * target_flat
                ).sum(dim=1)

                predicted_area = predicted_flat.sum(dim=1)

                dice = (
                    (2 * intersection + 1e-7)
                    / (
                        predicted_area
                        + target_area
                        + 1e-7
                    )
                )

                total_dice[threshold] += (
                    dice[lesion_images].sum().item()
                )

                normal_predictions = (
                    predicted_area[normal_images]
                )

                normal_fp[threshold] += (
                    (normal_predictions > 0).sum().item()
                )

                normal_predicted_pixels[threshold] += (
                    normal_predictions.sum().item()
                )

    total_samples = classification_confusion.sum().item()

    correct = classification_confusion.diag().sum().item()

    accuracy = correct / total_samples

    print("\n========== SEGMENTATION THRESHOLD EXPERIMENT ==========")
    print("Device:", device)
    print("Checkpoint:", CHECKPOINT.name)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation samples:", total_samples)
    print("Classification accuracy:", round(accuracy, 4))
    print("Lesion images:", lesion_count)
    print("Normal images:", normal_count)

    print("\nClassification confusion matrix:")
    print(classification_confusion.tolist())

    print("\nThreshold-wise segmentation results:")

    print(
        f"{'Threshold':>10}"
        f"{'Lesion Dice':>15}"
        f"{'Normal FP':>13}"
        f"{'FP Rate':>12}"
        f"{'Normal FP Pixels':>19}"
    )

    for threshold in THRESHOLDS:

        lesion_dice = (
            total_dice[threshold] / lesion_count
            if lesion_count else 0.0
        )

        fp_rate = (
            normal_fp[threshold] / normal_count
            if normal_count else 0.0
        )

        row = {
            "checkpoint": CHECKPOINT.name,
            "epoch": checkpoint["epoch"],
            "threshold": threshold,
            "samples": total_samples,
            "classification_accuracy": accuracy,
            "lesion_images": lesion_count,
            "lesion_dice": lesion_dice,
            "normal_images": normal_count,
            "normal_false_positives": normal_fp[threshold],
            "normal_fp_rate": fp_rate,
            "normal_predicted_mask_pixels": (
                normal_predicted_pixels[threshold]
            )
        }

        results.append(row)

        print(
            f"{threshold:>10.2f}"
            f"{lesion_dice:>15.4f}"
            f"{normal_fp[threshold]:>13}"
            f"{fp_rate:>12.4f}"
            f"{normal_predicted_pixels[threshold]:>19}"
        )

    columns = [
        "checkpoint",
        "epoch",
        "threshold",
        "samples",
        "classification_accuracy",
        "lesion_images",
        "lesion_dice",
        "normal_images",
        "normal_false_positives",
        "normal_fp_rate",
        "normal_predicted_mask_pixels"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:")
    print(OUTPUT_CSV)

    print("\nThreshold experiment completed.")


if __name__ == "__main__":
    main()
