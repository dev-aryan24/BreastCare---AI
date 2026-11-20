
import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (probabilities * targets).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


class MultiTaskLoss(nn.Module):

    def __init__(
        self,
        classification_weight=1.0,
        segmentation_weight=1.0
    ):
        super().__init__()

        self.classification_loss = nn.CrossEntropyLoss()
        self.segmentation_bce = nn.BCEWithLogitsLoss()
        self.segmentation_dice = DiceLoss()

        self.classification_weight = classification_weight
        self.segmentation_weight = segmentation_weight

    def forward(
        self,
        classification_logits,
        segmentation_logits,
        class_targets,
        mask_targets
    ):
        classification_loss = self.classification_loss(
            classification_logits,
            class_targets
        )

        bce_loss = self.segmentation_bce(
            segmentation_logits,
            mask_targets
        )

        dice_loss = self.segmentation_dice(
            segmentation_logits,
            mask_targets
        )

        segmentation_loss = bce_loss + dice_loss

        total_loss = (
            self.classification_weight * classification_loss
            + self.segmentation_weight * segmentation_loss
        )

        return {
            "total_loss": total_loss,
            "classification_loss": classification_loss,
            "segmentation_loss": segmentation_loss,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss
        }
