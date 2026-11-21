
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from src.data.preprocessing import preprocess_image
from src.models.multitask_cnn import MultiTaskCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "best_multitask_cnn_gpu.pth"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"

CLASS_NAMES = ["Benign", "Malignant", "Normal"]

IMAGE_SIZE = (224, 224)

SEGMENTATION_THRESHOLD = 0.70
MINIMUM_COMPONENT_AREA = 200
MINIMUM_MEAN_PROBABILITY = 0.80


def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MultiTaskCNN(num_classes=3).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def filter_segmentation_components(probability_map):
    original_mask = probability_map >= SEGMENTATION_THRESHOLD

    labeled_mask, component_count = ndimage.label(original_mask)

    filtered_mask = np.zeros_like(original_mask, dtype=bool)

    for component_id in range(1, component_count + 1):
        component = labeled_mask == component_id

        area = int(component.sum())

        if area < MINIMUM_COMPONENT_AREA:
            continue

        mean_probability = float(
            probability_map[component].mean()
        )

        if mean_probability >= MINIMUM_MEAN_PROBABILITY:
            filtered_mask |= component

    return filtered_mask


def create_overlay(image, mask):
    image_array = np.asarray(image).copy()

    overlay = image_array.copy()

    overlay[mask] = [255, 0, 0]

    blended = (
        image_array.astype(np.float32) * 0.65
        + overlay.astype(np.float32) * 0.35
    )

    return Image.fromarray(
        np.clip(blended, 0, 255).astype(np.uint8)
    )


def predict_image(image_path, apply_filter=False):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model, device = load_model()

    original_image = Image.open(image_path).convert("RGB")

    resized_image = original_image.resize(
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        Image.Resampling.BILINEAR
    )

    image_tensor = preprocess_image(
        image_path,
        image_size=IMAGE_SIZE
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        classification_logits, segmentation_logits = model(
            image_tensor
        )

        class_probabilities = torch.softmax(
            classification_logits,
            dim=1
        )[0].cpu().numpy()

        segmentation_probabilities = torch.sigmoid(
            segmentation_logits
        )[0, 0].cpu().numpy()

    predicted_class_index = int(
        np.argmax(class_probabilities)
    )

    predicted_class = CLASS_NAMES[predicted_class_index]

    class_confidence = float(
        class_probabilities[predicted_class_index]
    )

    original_mask = (
        segmentation_probabilities >= SEGMENTATION_THRESHOLD
    )

    filtered_mask = filter_segmentation_components(
        segmentation_probabilities
    )

    selected_mask = (
        filtered_mask if apply_filter else original_mask
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_stem = image_path.stem

    raw_mask_path = OUTPUT_DIR / f"{image_stem}_raw_mask.png"

    filtered_mask_path = (
        OUTPUT_DIR / f"{image_stem}_filtered_mask.png"
    )

    raw_overlay_path = (
        OUTPUT_DIR / f"{image_stem}_raw_overlay.png"
    )

    filtered_overlay_path = (
        OUTPUT_DIR / f"{image_stem}_filtered_overlay.png"
    )

    Image.fromarray(
        original_mask.astype(np.uint8) * 255
    ).save(raw_mask_path)

    Image.fromarray(
        filtered_mask.astype(np.uint8) * 255
    ).save(filtered_mask_path)

    create_overlay(
        resized_image,
        original_mask
    ).save(raw_overlay_path)

    create_overlay(
        resized_image,
        filtered_mask
    ).save(filtered_overlay_path)

    return {
        "filename": image_path.name,
        "predicted_class": predicted_class,
        "class_confidence": class_confidence,
        "class_probabilities": {
            CLASS_NAMES[i]: float(class_probabilities[i])
            for i in range(len(CLASS_NAMES))
        },
        "original_mask_pixels": int(original_mask.sum()),
        "filtered_mask_pixels": int(filtered_mask.sum()),
        "selected_mask_pixels": int(selected_mask.sum()),
        "filter_applied": apply_filter,
        "selected_mask": selected_mask,
        "original_mask_path": str(raw_mask_path),
        "filtered_mask_path": str(filtered_mask_path),
        "original_overlay_path": str(raw_overlay_path),
        "filtered_overlay_path": str(filtered_overlay_path),
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser(
        description="BreastCare-AI image inference"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the breast ultrasound image"
    )

    parser.add_argument(
        "--apply-filter",
        action="store_true",
        help="Use experimental component filtering"
    )

    args = parser.parse_args()

    result = predict_image(
        args.image,
        apply_filter=args.apply_filter
    )

    print("\n========== BREASTCARE-AI PREDICTION ==========")

    print(f"Image: {result['filename']}")
    print(f"Device: {result['device']}")

    print("\nClassification:")
    print(f"Predicted class: {result['predicted_class']}")
    print(
        f"Confidence: {result['class_confidence'] * 100:.2f}%"
    )

    print("\nClass probabilities:")

    for class_name, probability in result["class_probabilities"].items():
        print(f"{class_name}: {probability * 100:.2f}%")

    print("\nSegmentation:")
    print(f"Original mask pixels: {result['original_mask_pixels']}")
    print(f"Filtered mask pixels: {result['filtered_mask_pixels']}")
    print(f"Selected mask pixels: {result['selected_mask_pixels']}")

    print(
        "Selected output:",
        "Experimental filtered mask"
        if result["filter_applied"]
        else "Original model mask"
    )

    print("\nGenerated files:")

    print(result["original_mask_path"])
    print(result["filtered_mask_path"])
    print(result["original_overlay_path"])
    print(result["filtered_overlay_path"])

    print("\nResearch prototype only. Not for clinical diagnosis.")


if __name__ == "__main__":
    main()
