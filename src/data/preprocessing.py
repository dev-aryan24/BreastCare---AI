
from PIL import Image
import numpy as np
import torch


def preprocess_image(image_path, image_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(
        (image_size[1], image_size[0]),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(image, dtype=np.float32) / 255.0
    image = (image - 0.5) / 0.5

    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image, dtype=torch.float32)


def preprocess_mask(mask_paths, image_size=(224, 224)):
    combined_mask = np.zeros(image_size, dtype=np.uint8)

    for mask_path in mask_paths:
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(
            (image_size[1], image_size[0]),
            Image.Resampling.NEAREST
        )

        mask_array = np.asarray(mask)
        combined_mask = np.maximum(
            combined_mask,
            (mask_array > 0).astype(np.uint8)
        )

    return torch.tensor(
        combined_mask,
        dtype=torch.float32
    ).unsqueeze(0)
