from src.data.busi_dataset import create_dataloaders

manifest = "data/processed/ultrasound/busi_manifest.csv"

train_loader, val_loader, test_loader = create_dataloaders(
    manifest,
    batch_size=4
)

batch = next(iter(train_loader))

print("Image shape:", batch["image"].shape)
print("Mask shape:", batch["mask"].shape)
print("Label shape:", batch["label"].shape)
print("Labels:", batch["label"])