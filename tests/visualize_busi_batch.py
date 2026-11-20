
import matplotlib.pyplot as plt
import torch

from src.data.busi_dataset import create_dataloaders

manifest = "data/processed/ultrasound/busi_manifest.csv"

train_loader, _, _ = create_dataloaders(
    manifest,
    batch_size=4
)

batch = next(iter(train_loader))

images = batch["image"]
masks = batch["mask"]
labels = batch["label"]

class_names = ["Benign", "Malignant", "Normal"]

fig, axes = plt.subplots(4, 3, figsize=(12, 14))

for i in range(4):
    image = images[i].permute(1, 2, 0).numpy()
    image = (image * 0.5) + 0.5
    image = image.clip(0, 1)

    mask = masks[i][0].numpy()

    axes[i, 0].imshow(image)
    axes[i, 0].set_title(class_names[labels[i].item()])
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("Segmentation Mask")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(image)
    axes[i, 2].imshow(mask, cmap="jet", alpha=0.4)
    axes[i, 2].set_title("Image + Mask Overlay")
    axes[i, 2].axis("off")
    
    axes[i, 2].imshow(image)
    axes[i, 2].imshow(mask, cmap="jet", alpha=0.4)
    axes[i, 2].set_title("Image + Mask Overlay")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig("data/processed/ultrasound/busi_batch_visualization.png")
plt.show()
