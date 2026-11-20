
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MultiTaskCNN(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.encoder1 = ConvBlock(3, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        self.encoder4 = ConvBlock(128, 256)

        self.bottleneck = ConvBlock(256, 512)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        self.decoder4 = ConvBlock(512 + 256, 256)
        self.decoder3 = ConvBlock(256 + 128, 128)
        self.decoder2 = ConvBlock(128 + 64, 64)
        self.decoder1 = ConvBlock(64 + 32, 32)

        self.segmentation_head = nn.Conv2d(
            32, 1, kernel_size=1
        )

    def forward(self, x):

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))

        classification = self.classifier(bottleneck)

        d4 = F.interpolate(
            bottleneck,
            size=e4.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(
            d4,
            size=e3.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(
            d3,
            size=e2.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(
            d2,
            size=e1.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        segmentation = self.segmentation_head(d1)

        return classification, segmentation
