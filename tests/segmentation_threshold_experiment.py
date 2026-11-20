
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.busi_dataset import create_dataloaders
from src.models.multitask_cnn import MultiTaskCNN


MANIFEST = PROJECT_ROOT / "data/processed/ultrasound/busi_manifest.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints/best_multitask_cnn_gpu.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs/threshold_analysis"

SWEEP_CSV = OUTPUT_DIR / "epoch12_threshold_sweep.csv"
AUDIT_CSV = OUTPUT_DIR / "normal_validation_fp_audit.csv"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
AUDIT_THRESHOLDS = [0.50, 0.65, 0.70]

CLASS_NAMES = ["Benign", "Malignant", "Normal"]


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    confusion = torch.zeros(3, 3, dtype=torch.int64)

    lesion_dice = {t: [] for t in THRESHOLDS}
    normal_fp_count = {t: 0 for t in THRESHOLDS}
    normal_fp_pixels = {t: 0 for t in THRESHOLDS}

    normal_audit = []

    total_samples = 0
    lesion_images = 0
    normal_images = 0
    classification_correct = 0

    with torch.inference_mode():

        for batch in val_loader:

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            class_logits, segmentation_logits = model(images)

            predicted_classes = class_logits.argmax(dim=1)
            probabilities = torch.sigmoid(segmentation_logits)

            total_samples += labels.size(0)
            classification_correct += (
                predicted_classes == labels
            ).sum().item()

            for actual, predicted in zip(
                labels.cpu(), predicted_classes.cpu()
            ):
                confusion[actual.item(), predicted.item()] += 1

            for i in range(len(labels)):

                actual_label = labels[i].item()
                filename = Path(batch["image_path"][i]).name

                target_mask = masks[i] > 0
                target_area = target_mask.sum().item()

                if actual_label in [0, 1]:
                    lesion_images += 1

                    if target_area > 0:
                        for threshold in THRESHOLDS:

                            predicted_mask = (
                                probabilities[i] >= threshold
                            )

                            intersection = (
                                predicted_mask & target_mask
                            ).sum().item()

                            predicted_area = (
                                predicted_mask.sum().item()
                            )

                            dice = (
                                2 * intersection + 1e-7
                            ) / (
                                predicted_area + target_area + 1e-7
                            )

                            lesion_dice[threshold].append(dice)

                elif actual_label == 2:

                    normal_images += 1

                    audit_row = {
                        "filename": filename,
                        "predicted_class": CLASS_NAMES[
                            predicted_classes[i].item()
                        ],
                        "classification_confidence": round(
                            torch.softmax(
                                class_logits[i], dim=0
                            ).max().item(),
                            6
                        )
                    }

                    for threshold in THRESHOLDS:

                        predicted_mask = (
                            probabilities[i] >= threshold
                        )

                        pixel_count = predicted_mask.sum().item()

                        normal_fp_count[threshold] += (
                            pixel_count > 0
                        )

                        normal_fp_pixels[threshold] += pixel_count

                        if threshold in AUDIT_THRESHOLDS:

                            audit_row[
                                f"pixels_t{threshold:.2f}"
                            ] = pixel_count

                            audit_row[
                                f"false_positive_t{threshold:.2f}"
                            ] = pixel_count > 0

                    normal_audit.append(audit_row)

    accuracy = classification_correct / total_samples

    print("\n========== SEGMENTATION THRESHOLD EXPERIMENT ==========")
    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation samples:", total_samples)
    print("Classification accuracy:", round(accuracy, 4))
    print("Lesion images:", lesion_images)
    print("Normal images:", normal_images)

    print("\nClassification confusion matrix:")
    print(confusion.tolist())

    print("\nThreshold-wise results:")
    print(
        f"{'Threshold':>10} {'Lesion Dice':>14} "
        f"{'Normal FP':>12} {'FP Rate':>10} {'FP Pixels':>12}"
    )

    sweep_rows = []

    for threshold in THRESHOLDS:

        dice_value = (
            sum(lesion_dice[threshold])
            / len(lesion_dice[threshold])
            if lesion_dice[threshold]
            else 0
        )

        fp_count = normal_fp_count[threshold]
        fp_rate = fp_count / normal_images if normal_images else 0
        fp_pixels = normal_fp_pixels[threshold]

        print(
            f"{threshold:>10.2f} "
            f"{dice_value:>14.4f} "
            f"{fp_count:>12} "
            f"{fp_rate:>10.4f} "
            f"{fp_pixels:>12}"
        )

        sweep_rows.append({
            "threshold": threshold,
            "lesion_dice": round(dice_value, 6),
            "normal_fp_images": fp_count,
            "normal_fp_rate": round(fp_rate, 6),
            "normal_predicted_pixels": fp_pixels
        })

    with open(SWEEP_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=sweep_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(sweep_rows)

    audit_columns = [
        "filename",
        "predicted_class",
        "classification_confidence",
        "pixels_t0.50",
        "false_positive_t0.50",
        "pixels_t0.65",
        "false_positive_t0.65",
        "pixels_t0.70",
        "false_positive_t0.70"
    ]

    normal_audit.sort(key=lambda row: row["filename"])

    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=audit_columns
        )
        writer.writeheader()
        writer.writerows(normal_audit)

    print("\n========== NORMAL IMAGE AUDIT ==========")

    for threshold in AUDIT_THRESHOLDS:

        count = normal_fp_count[threshold]

        print(
            f"Threshold {threshold:.2f}: "
            f"{count}/{normal_images} images with predicted masks"
        )

    print("\nTop normal-image false-positive masks at threshold 0.50:")

    top_images = sorted(
        normal_audit,
        key=lambda row: row["pixels_t0.50"],
        reverse=True
    )

    for row in top_images[:5]:
        print(
            row["filename"],
            "| Pixels:",
            row["pixels_t0.50"]
        )

    print("\nResults saved:")
    print(SWEEP_CSV)
    print(AUDIT_CSV)

    print("\nThreshold experiment and normal-image audit completed.")


if __name__ == "__main__":
    main()
