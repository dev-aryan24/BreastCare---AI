
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.busi_dataset import create_dataloaders
from src.models.multitask_cnn import MultiTaskCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    PROJECT_ROOT
    / "data/processed/ultrasound/busi_manifest.csv"
)

CHECKPOINTS = [
    PROJECT_ROOT / "checkpoints/best_multitask_cnn.pth",
    PROJECT_ROOT / "checkpoints/best_multitask_cnn_gpu.pth",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs/checkpoint_comparison"
OUTPUT_CSV = OUTPUT_DIR / "validation_comparison.csv"

CLASS_NAMES = ["Benign", "Malignant", "Normal"]


def evaluate_checkpoint(checkpoint_path, model, val_loader, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion = torch.zeros(3, 3, dtype=torch.int64)

    total_dice = 0.0
    lesion_count = 0
    missing_lesion_masks = 0

    normal_false_positives = 0
    normal_count = 0

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
                confusion[actual, predicted] += 1

            predicted_masks = (
                torch.sigmoid(segmentation_logits) >= 0.5
            ).float()

            predicted_flat = predicted_masks.flatten(1)
            target_flat = masks.flatten(1)

            intersection = (
                predicted_flat * target_flat
            ).sum(dim=1)

            predicted_area = predicted_flat.sum(dim=1)
            target_area = target_flat.sum(dim=1)

            dice = (
                (2 * intersection + 1e-7)
                / (predicted_area + target_area + 1e-7)
            )

            lesion_images = labels != 2
            normal_images = labels == 2

            valid_lesion_images = (
                lesion_images & (target_area > 0)
            )

            missing_lesion_masks += (
                lesion_images & (target_area == 0)
            ).sum().item()

            total_dice += dice[
                valid_lesion_images
            ].sum().item()

            lesion_count += (
                valid_lesion_images.sum().item()
            )

            normal_false_positives += (
                (predicted_area[normal_images] > 0)
                .sum()
                .item()
            )

            normal_count += normal_images.sum().item()

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()

    accuracy = correct / total if total else 0.0

    class_metrics = {}

    for index, name in enumerate(CLASS_NAMES):

        tp = confusion[index, index].item()
        fp = confusion[:, index].sum().item() - tp
        fn = confusion[index, :].sum().item() - tp

        precision = (
            tp / (tp + fp)
            if tp + fp else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        class_metrics[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    macro_f1 = sum(
        values["f1"]
        for values in class_metrics.values()
    ) / len(CLASS_NAMES)

    lesion_dice = (
        total_dice / lesion_count
        if lesion_count else 0.0
    )

    normal_fp_rate = (
        normal_false_positives / normal_count
        if normal_count else 0.0
    )

    return {
        "checkpoint": checkpoint_path.name,
        "epoch": checkpoint["epoch"],
        "stored_val_loss": checkpoint.get("val_loss"),
        "samples": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "lesion_dice": lesion_dice,
        "lesion_images": lesion_count,
        "missing_lesion_masks": missing_lesion_masks,
        "normal_fp_rate": normal_fp_rate,
        "normal_false_positives": normal_false_positives,
        "normal_images": normal_count,
        "confusion": confusion.tolist(),
        "class_metrics": class_metrics
    }


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

    results = []

    print("\n========== VALIDATION CHECKPOINT COMPARISON ==========")
    print("Device:", device)
    print("Validation samples:", len(val_loader.dataset))

    for checkpoint_path in CHECKPOINTS:

        print("\nEvaluating:", checkpoint_path.name)

        if not checkpoint_path.exists():
            print("Checkpoint not found. Skipping.")
            continue

        result = evaluate_checkpoint(
            checkpoint_path,
            model,
            val_loader,
            device
        )

        results.append(result)

        print("\nEpoch:", result["epoch"])
        print("Stored validation loss:", result["stored_val_loss"])
        print("Samples:", result["samples"])
        print("Accuracy:", round(result["accuracy"], 4))
        print("Macro F1:", round(result["macro_f1"], 4))
        print("Lesion Dice:", round(result["lesion_dice"], 4))
        print("Normal FP rate:", round(result["normal_fp_rate"], 4))

        print("\nConfusion Matrix")
        print("Rows = Actual | Columns = Predicted")
        print("             Benign Malignant Normal")

        for name, row in zip(
            CLASS_NAMES,
            result["confusion"]
        ):
            print(f"{name:12}", row)

        print("\nPer-Class Metrics")

        for name, metrics in result["class_metrics"].items():
            print(
                f"{name}: "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}"
            )

        print(
            "Missing lesion masks:",
            result["missing_lesion_masks"]
        )

    if not results:
        print("\nNo checkpoints were evaluated.")
        return

    csv_columns = [
        "checkpoint",
        "epoch",
        "stored_val_loss",
        "samples",
        "accuracy",
        "macro_f1",
        "lesion_dice",
        "lesion_images",
        "missing_lesion_masks",
        "normal_fp_rate",
        "normal_false_positives",
        "normal_images"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_columns
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                key: result[key]
                for key in csv_columns
            })

    print("\n========== COMPARISON SUMMARY ==========")

    print(
        f"{'Checkpoint':32}"
        f"{'Epoch':>7}"
        f"{'Accuracy':>12}"
        f"{'Macro F1':>12}"
        f"{'Dice':>10}"
        f"{'Normal FP':>12}"
    )

    for result in results:

        print(
            f"{result['checkpoint']:32}"
            f"{result['epoch']:>7}"
            f"{result['accuracy']:>12.4f}"
            f"{result['macro_f1']:>12.4f}"
            f"{result['lesion_dice']:>10.4f}"
            f"{result['normal_fp_rate']:>12.4f}"
        )

    print("\nResults saved to:")
    print(OUTPUT_CSV)

    print("\nValidation comparison completed.")


if __name__ == "__main__":
    main()
