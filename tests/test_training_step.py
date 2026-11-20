
import torch

from src.data.busi_dataset import create_dataloaders
from src.models.multitask_cnn import MultiTaskCNN
from src.models.losses import MultiTaskLoss

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

manifest = "data/processed/ultrasound/busi_manifest.csv"

train_loader, _, _ = create_dataloaders(
    manifest,
    batch_size=2
)

model = MultiTaskCNN(num_classes=3).to(device)
criterion = MultiTaskLoss().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

batch = next(iter(train_loader))

images = batch["image"].to(device)
masks = batch["mask"].to(device)
labels = batch["label"].to(device)

model.train()
optimizer.zero_grad()

classification_logits, segmentation_logits = model(images)

losses = criterion(
    classification_logits,
    segmentation_logits,
    labels,
    masks
)

losses["total_loss"].backward()
optimizer.step()

print("Device:", device)
print("Total loss:", losses["total_loss"].item())
print("Classification loss:", losses["classification_loss"].item())
print("Segmentation loss:", losses["segmentation_loss"].item())
print("BCE loss:", losses["bce_loss"].item())
print("Dice loss:", losses["dice_loss"].item())

print("\nBackward propagation: successful")
print("Optimizer step: successful")

