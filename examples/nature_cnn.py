"""CIFAR-sized baseline and modular CNN architectures with shape checks."""

from __future__ import annotations

import torch
from torch import nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NatureCNN(nn.Module):
    def __init__(self, classes: int = 15) -> None:
        super().__init__()
        self.features = nn.Sequential(
            CNNBlock(3, 32),
            CNNBlock(32, 64),
            CNNBlock(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CompactNatureCNN(nn.Module):
    def __init__(self, classes: int = 15) -> None:
        super().__init__()
        self.features = nn.Sequential(
            CNNBlock(3, 32),
            CNNBlock(32, 64),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def smoke_test() -> dict[str, int]:
    torch.manual_seed(42)
    batch = torch.randn(4, 3, 32, 32)
    baseline = NatureCNN()
    compact = CompactNatureCNN()
    assert baseline(batch).shape == (4, 15)
    assert compact(batch).shape == (4, 15)
    return {
        "baseline_parameters": sum(parameter.numel() for parameter in baseline.parameters()),
        "compact_parameters": sum(parameter.numel() for parameter in compact.parameters()),
    }


if __name__ == "__main__":
    print(smoke_test())

