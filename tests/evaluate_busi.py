
import torch
from pathlib import Path

from src.data.busi_dataset import create_dataloaders
from src.models.multitask_cnn import MultiTaskCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = PROJECT_ROOT / "data/processed/ultrasound/busi_manifest.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints/best_multitask_cnn_gpu.pth"

CLASS_NAMES = ["Benign", "Malignant", "Normal"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = create_dataloaders(
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

    total_dice = 0.0
    lesion_count = 0
    normal_false_positives = 0
    normal_count = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            class_logits, segmentation_logits = model(images)

            predictions = class_logits.argmax(dim=1)

            for actual, predicted in zip(labels.cpu(), predictions.cpu()):
                confusion[actual, predicted] += 1

            predicted_masks = (
                torch.sigmoid(segmentation_logits) >= 0.5
            ).float()

            predicted_flat = predicted_masks.flatten(1)
            target_flat = masks.flatten(1)

            intersection = (predicted_flat * target_flat).sum(dim=1)
            predicted_area = predicted_flat.sum(dim=1)
            target_area = target_flat.sum(dim=1)

            dice = (
                (2 * intersection + 1e-7) /
                (predicted_area + target_area + 1e-7)
            )

            lesion_images = target_area > 0
            normal_images = ~lesion_images

            total_dice += dice[lesion_images].sum().item()
            lesion_count += lesion_images.sum().item()

            normal_false_positives += (
                (predicted_area[normal_images] > 0).sum().item()
            )
            normal_count += normal_images.sum().item()

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()

    print("\n========== TEST SET EVALUATION ==========")
    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Test samples:", total)
    print("Accuracy:", round(correct / total, 4))

    print("\nConfusion Matrix")
    print("Rows = Actual | Columns = Predicted")
    print("             Benign Malignant Normal")

    for name, row in zip(CLASS_NAMES, confusion.tolist()):
        print(f"{name:12}", row)

    print("\nPer-Class Classification Metrics")

    for index, name in enumerate(CLASS_NAMES):
        tp = confusion[index, index].item()
        fp = confusion[:, index].sum().item() - tp
        fn = confusion[index, :].sum().item() - tp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        print(
            f"{name}: "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

    print("\nSegmentation Metrics")

    print(
        "Lesion Dice:",
        round(total_dice / lesion_count, 4)
        if lesion_count else "N/A"
    )

    print("Lesion images:", lesion_count)

    print(
        "Normal false-positive rate:",
        round(normal_false_positives / normal_count, 4)
        if normal_count else "N/A"
    )

    print("Normal images:", normal_count)
    print("\nTest evaluation completed.")


if __name__ == "__main__":
    main()
