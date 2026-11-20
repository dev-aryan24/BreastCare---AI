
import csv
import random
from pathlib import Path

import numpy as np
import torch

from src.data.busi_dataset import create_dataloaders
from src.models.multitask_cnn import MultiTaskCNN
from src.models.losses import MultiTaskLoss


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = PROJECT_ROOT / "data/processed/ultrasound/busi_manifest.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INITIAL_CHECKPOINT = CHECKPOINT_DIR / "best_multitask_cnn.pth"
GPU_CHECKPOINT = CHECKPOINT_DIR / "best_multitask_cnn_gpu.pth"
HISTORY_PATH = OUTPUT_DIR / "gpu_training_history.csv"

EPOCHS = 20
BATCH_SIZE = 2
LEARNING_RATE = 0.0001
PATIENCE = 5
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(class_logits, segmentation_logits, labels, masks):
    predictions = class_logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()

    predicted_masks = (
        torch.sigmoid(segmentation_logits) >= 0.5
    ).float()

    predicted_flat = predicted_masks.flatten(start_dim=1)
    target_flat = masks.flatten(start_dim=1)

    target_areas = target_flat.sum(dim=1)
    predicted_areas = predicted_flat.sum(dim=1)

    lesion_images = target_areas > 0
    normal_images = ~lesion_images

    intersection = (predicted_flat * target_flat).sum(dim=1)

    dice_values = (
        (2 * intersection + 1e-7) /
        (predicted_areas + target_areas + 1e-7)
    )

    lesion_count = lesion_images.sum().item()
    normal_count = normal_images.sum().item()

    lesion_dice_sum = (
        dice_values[lesion_images].sum().item()
        if lesion_count > 0 else 0.0
    )

    false_positive_count = (
        (predicted_areas[normal_images] > 0).sum().item()
        if normal_count > 0 else 0
    )

    return {
        "correct": correct,
        "samples": labels.size(0),
        "lesion_dice_sum": lesion_dice_sum,
        "lesion_count": lesion_count,
        "false_positive_count": false_positive_count,
        "normal_count": normal_count
    }


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    total_lesion_dice = 0.0
    total_lesion_images = 0

    total_false_positives = 0
    total_normal_images = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            class_logits, segmentation_logits = model(images)

            losses = criterion(
                class_logits,
                segmentation_logits,
                labels,
                masks
            )

            if training:
                losses["total_loss"].backward()
                optimizer.step()

        metrics = calculate_metrics(
            class_logits.detach(),
            segmentation_logits.detach(),
            labels,
            masks
        )

        batch_size = images.size(0)

        total_loss += losses["total_loss"].item() * batch_size
        total_correct += metrics["correct"]
        total_samples += metrics["samples"]

        total_lesion_dice += metrics["lesion_dice_sum"]
        total_lesion_images += metrics["lesion_count"]

        total_false_positives += metrics["false_positive_count"]
        total_normal_images += metrics["normal_count"]

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "lesion_dice": (
            total_lesion_dice / total_lesion_images
            if total_lesion_images else None
        ),
        "normal_false_positive_rate": (
            total_false_positives / total_normal_images
            if total_normal_images else None
        )
    }


def format_metric(value):
    return f"{value:.4f}" if value is not None else "N/A"


def save_history(history):
    with HISTORY_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )
        writer.writeheader()
        writer.writerows(history)


def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {MANIFEST}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader, _ = create_dataloaders(
        str(MANIFEST),
        batch_size=BATCH_SIZE
    )

    model = MultiTaskCNN(num_classes=3).to(device)
    criterion = MultiTaskLoss().to(device)

    if INITIAL_CHECKPOINT.exists():
        checkpoint = torch.load(
            INITIAL_CHECKPOINT,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        print("Loaded existing CPU checkpoint.")
        print("Starting from epoch:", checkpoint["epoch"])
    else:
        print("No initial checkpoint found. Starting from scratch.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    print("\n========== GPU TRAINING ==========")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU Memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3, 2
            ),
            "GB"
        )

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", EPOCHS)

    for epoch in range(EPOCHS):
        train_metrics = run_epoch(
            model, train_loader, criterion,
            optimizer, device, training=True
        )

        val_metrics = run_epoch(
            model, val_loader, criterion,
            optimizer, device, training=False
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_lesion_dice": train_metrics["lesion_dice"],
            "train_normal_false_positive_rate": train_metrics["normal_false_positive_rate"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_lesion_dice": val_metrics["lesion_dice"],
            "val_normal_false_positive_rate": val_metrics["normal_false_positive_rate"]
        }

        history.append(row)
        save_history(history)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(
            f"Train Loss: {row['train_loss']:.4f} | "
            f"Accuracy: {row['train_accuracy']:.4f} | "
            f"Lesion Dice: {format_metric(row['train_lesion_dice'])} | "
            f"Normal FP Rate: {format_metric(row['train_normal_false_positive_rate'])}"
        )
        print(
            f"Val Loss: {row['val_loss']:.4f} | "
            f"Accuracy: {row['val_accuracy']:.4f} | "
            f"Lesion Dice: {format_metric(row['val_lesion_dice'])} | "
            f"Normal FP Rate: {format_metric(row['val_normal_false_positive_rate'])}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "training_config": {
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "seed": SEED,
                        "initial_checkpoint": str(INITIAL_CHECKPOINT)
                    }
                },
                GPU_CHECKPOINT
            )

            print("Best GPU checkpoint saved.")

        else:
            epochs_without_improvement += 1
            print(
                "No validation improvement:",
                epochs_without_improvement,
                "/",
                PATIENCE
            )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\n========== TRAINING COMPLETED ==========")
    print("Best validation loss:", round(best_val_loss, 4))
    print("GPU checkpoint:", GPU_CHECKPOINT)
    print("Training history:", HISTORY_PATH)


if __name__ == "__main__":
    main()
