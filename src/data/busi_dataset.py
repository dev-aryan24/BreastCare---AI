
import csv
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.preprocessing import (
    preprocess_image,
    preprocess_mask
)


class BUSIDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split,
        image_size=(224, 224)
    ):
        self.image_size = image_size
        self.records = []

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["split"] == split:
                    self.records.append(row)

        self.class_to_index = {
            "benign": 0,
            "malignant": 1,
            "normal": 2
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image = preprocess_image(
            record["image_path"],
            self.image_size
        )

        mask_paths = (
            record["mask_paths"].split(";")
            if record["mask_paths"]
            else []
        )

        mask = preprocess_mask(
            mask_paths,
            self.image_size
        )

        label = self.class_to_index[record["class"]]

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(
                label,
                dtype=torch.long
            ),
            "image_path": record["image_path"],
            "class_name": record["class"]
        }


def create_dataloaders(
    manifest_path,
    batch_size=16
):
    train_dataset = BUSIDataset(
        manifest_path,
        "train"
    )

    val_dataset = BUSIDataset(
        manifest_path,
        "validation"
    )

    test_dataset = BUSIDataset(
        manifest_path,
        "test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )
