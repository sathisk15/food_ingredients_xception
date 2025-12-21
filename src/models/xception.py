import torch
import torch.nn as nn
from .layers import SeparableConv2d


class XceptionBlock(nn.Module):
    """
    Xception residual block
    """

    def __init__(self, in_channels, out_channels, reps, stride=1):
        super().__init__()

        self.skip = None
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        for i in range(reps):
            layers.append(nn.ReLU(inplace=False))
            layers.append(
                SeparableConv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

        self.sepconvs = nn.Sequential(*layers)

        self.pool = (
            nn.MaxPool2d(3, stride=stride, padding=1)
            if stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        identity = x

        x = self.sepconvs(x)
        x = self.pool(x)

        if self.skip is not None:
            identity = self.skip(identity)

        return x + identity


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Entry flow
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )

        self.block1 = XceptionBlock(64, 128, reps=2, stride=2)
        self.block2 = XceptionBlock(128, 256, reps=2, stride=2)
        self.block3 = XceptionBlock(256, 728, reps=2, stride=2)

        # Middle flow (8 identical blocks)
        self.middle_blocks = nn.Sequential(
            *[XceptionBlock(728, 728, reps=3, stride=1) for _ in range(8)]
        )

        # Exit flow
        self.block_exit = XceptionBlock(728, 1024, reps=2, stride=2)

        self.conv_exit = nn.Sequential(
            nn.ReLU(inplace=False),
            SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.middle_blocks(x)

        x = self.block_exit(x)
        x = self.conv_exit(x)

        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x
