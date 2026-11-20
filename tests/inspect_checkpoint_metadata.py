
from pathlib import Path
import torch

CHECKPOINTS = [
    Path("checkpoints/best_multitask_cnn.pth"),
    Path("checkpoints/best_multitask_cnn_gpu.pth"),
]

for path in CHECKPOINTS:
    print("\n" + "=" * 55)
    print("FILE:", path.name)
    print("=" * 55)

    if not path.exists():
        print("Checkpoint not found.")
        continue

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if not isinstance(checkpoint, dict):
        print("Unexpected checkpoint format:", type(checkpoint))
        continue

    print("Keys:", list(checkpoint.keys()))

    for key in [
        "epoch",
        "val_loss",
        "training_config"
    ]:
        print(f"{key}:", checkpoint.get(key, "Not available"))

    if "model_state_dict" in checkpoint:
        print(
            "Model parameter tensors:",
            len(checkpoint["model_state_dict"])
        )

    print("File size (MB):", round(path.stat().st_size / (1024 ** 2), 2))
