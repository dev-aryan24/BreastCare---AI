
import torch

from src.models.multitask_cnn import MultiTaskCNN

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MultiTaskCNN(num_classes=3).to(device)

dummy_input = torch.randn(4, 3, 224, 224).to(device)

classification, segmentation = model(dummy_input)

print("Device:", device)
print("Classification output:", classification.shape)
print("Segmentation output:", segmentation.shape)
print("Model parameters:", sum(
    p.numel() for p in model.parameters()
))

print("\nArchitecture test completed.")
