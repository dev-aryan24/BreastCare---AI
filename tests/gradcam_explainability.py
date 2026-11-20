
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data.busi_dataset import BUSIDataset
from src.models.multitask_cnn import MultiTaskCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = PROJECT_ROOT / "data/processed/ultrasound/busi_manifest.csv"

CHECKPOINT = PROJECT_ROOT / "checkpoints/best_multitask_cnn_gpu.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs/gradcam_analysis"
IMAGE_DIR = OUTPUT_DIR / "images"

CLASS_NAMES = ["Benign", "Malignant", "Normal"]

TARGET_IMAGES = [
    "malignant (111).png",
    "malignant (28).png",
    "malignant (48).png",
    "normal (117).png",
    "normal (111).png",
    "benign (256).png"
]


def normalize_cam(cam):
    cam = cam - cam.min()
    maximum = cam.max()

    if maximum > 0:
        cam = cam / maximum

    return cam


def generate_gradcam(
    model,
    image_tensor,
    target_class,
    captured
):
    model.zero_grad(set_to_none=True)
    captured.clear()

    class_logits, _ = model(image_tensor)
    score = class_logits[0, target_class]

    score.backward()

    activations = captured["activations"].detach()
    gradients = captured["gradients"].detach()

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (weights * activations).sum(
        dim=1,
        keepdim=True
    )

    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=image_tensor.shape[2:],
        mode="bilinear",
        align_corners=False
    )

    cam = cam[0, 0].cpu().numpy()

    return normalize_cam(cam)


def save_gradcam_visualization(
    image,
    predicted_cam,
    actual_cam,
    filename,
    actual_class,
    predicted_class,
    confidence,
    output_path
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Ultrasound")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(
        predicted_cam,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    axes[1].set_title(
        f"Predicted Class: {CLASS_NAMES[predicted_class]}"
    )

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(
        actual_cam,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    axes[2].set_title(
        f"Actual Class: {CLASS_NAMES[actual_class]}"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{filename}\n"
        f"Actual: {CLASS_NAMES[actual_class]} | "
        f"Predicted: {CLASS_NAMES[predicted_class]} | "
        f"Confidence: {confidence:.4f}"
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    dataset = BUSIDataset(
        str(MANIFEST),
        split="test"
    )

    model = MultiTaskCNN(
        num_classes=3
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    target_layer = model.bottleneck.block[-1]

    captured = {}

    def forward_hook(module, inputs, output):

        captured["activations"] = output

        output.register_hook(
            lambda gradient: captured.__setitem__(
                "gradients",
                gradient
            )
        )

    hook_handle = target_layer.register_forward_hook(
        forward_hook
    )

    records = []

    try:

        for filename in TARGET_IMAGES:

            matching_indices = [
                index
                for index, record in enumerate(dataset.records)
                if Path(record["image_path"]).name == filename
            ]

            if not matching_indices:
                print(f"Image not found: {filename}")
                continue

            sample = dataset[matching_indices[0]]

            image_tensor = (
                sample["image"].unsqueeze(0).to(device)
            )

            actual_class = sample["label"].item()

            with torch.enable_grad():

                logits, _ = model(image_tensor)

                probabilities = torch.softmax(
                    logits.detach(),
                    dim=1
                )

                confidence, prediction = probabilities.max(
                    dim=1
                )

            predicted_class = prediction.item()
            confidence_value = confidence.item()

            predicted_cam = generate_gradcam(
                model,
                image_tensor,
                predicted_class,
                captured
            )

            actual_cam = generate_gradcam(
                model,
                image_tensor,
                actual_class,
                captured
            )

            image = sample["image"].cpu().numpy()

            image = np.transpose(
                image,
                (1, 2, 0)
            )

            image = np.clip(
                image * 0.5 + 0.5,
                0,
                1
            )

            image = image.mean(axis=2)

            output_path = (
                IMAGE_DIR
                / f"{Path(filename).stem}_gradcam.png"
            )

            save_gradcam_visualization(
                image,
                predicted_cam,
                actual_cam,
                filename,
                actual_class,
                predicted_class,
                confidence_value,
                output_path
            )

            records.append({
                "filename": filename,
                "actual_class": CLASS_NAMES[actual_class],
                "predicted_class": CLASS_NAMES[predicted_class],
                "confidence": round(confidence_value, 6),
                "classification_correct": (
                    actual_class == predicted_class
                ),
                "predicted_class_gradcam": str(output_path)
            })

            print(f"\nImage: {filename}")
            print("Actual class:", CLASS_NAMES[actual_class])
            print("Predicted class:", CLASS_NAMES[predicted_class])
            print("Confidence:", round(confidence_value, 4))
            print("Grad-CAM saved:", output_path)

    finally:
        hook_handle.remove()

    csv_path = OUTPUT_DIR / "gradcam_results.csv"

    if records:

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=records[0].keys()
            )

            writer.writeheader()
            writer.writerows(records)

    print("\n========== GRAD-CAM COMPLETED ==========")
    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Images processed:", len(records))
    print("CSV report:", csv_path)
    print("Visualization directory:", IMAGE_DIR)


if __name__ == "__main__":
    main()
