"""Independent EMNIST baseline and CNN architectures plus shape smoke tests."""

from __future__ import annotations

import torch
from torch import nn


class DenseLetterClassifier(nn.Module):
    def __init__(self, classes: int = 26) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ConvolutionalLetterClassifier(nn.Module):
    def __init__(self, classes: int = 26) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(1)
        return self.classifier(features)


def smoke_test() -> dict[str, int]:
    torch.manual_seed(42)
    batch = torch.randn(8, 1, 28, 28)
    dense = DenseLetterClassifier()
    cnn = ConvolutionalLetterClassifier()
    dense_output = dense(batch)
    cnn_output = cnn(batch)
    assert dense_output.shape == (8, 26)
    assert cnn_output.shape == (8, 26)
    return {
        "dense_parameters": sum(parameter.numel() for parameter in dense.parameters()),
        "cnn_parameters": sum(parameter.numel() for parameter in cnn.parameters()),
    }


if __name__ == "__main__":
    print(smoke_test())

