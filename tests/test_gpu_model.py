
import torch
from src.models.multitask_cnn import MultiTaskCNN
from src.models.losses import MultiTaskLoss

device = torch.device("cuda")

model = MultiTaskCNN(num_classes=3).to(device)
criterion = MultiTaskLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

images = torch.randn(2, 3, 224, 224, device=device)
masks = torch.randint(0, 2, (2, 1, 224, 224)).float().to(device)
labels = torch.tensor([0, 1], device=device)

class_logits, segmentation_logits = model(images)

losses = criterion(class_logits, segmentation_logits, labels, masks)

optimizer.zero_grad()
losses["total_loss"].backward()
optimizer.step()

torch.cuda.synchronize()

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))
print("Classification output:", class_logits.shape)
print("Segmentation output:", segmentation_logits.shape)
print("Total loss:", losses["total_loss"].item())
print("GPU allocated memory:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")
print("GPU model test: PASSED")
