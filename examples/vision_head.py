"""Head-only ResNet training with noise; offline demo or your ImageFolder data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from recall_patterns import ImpulseNoise


class ToyImages(Dataset):
    """Original synthetic inputs to check mechanics, not transfer quality."""
    def __init__(self, size, transform, seed):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, 3, 32, 32, generator=generator)
        self.labels = torch.arange(size) % 3
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]


def run(output: Path, data=None, pretrained=False, epochs=2, noise=0.02):
    torch.set_num_threads(2)
    torch.manual_seed(31)
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    if pretrained and data is None:
        raise ValueError("--pretrained needs --data with train/ and val/ class folders")
    if data is None:
        train_data = ToyImages(24, transforms.Compose([
            transforms.RandomHorizontalFlip(), ImpulseNoise(noise)]), 41)
        val_data = ToyImages(12, transforms.Lambda(lambda image: image.clone()), 42)
        classes = ["synthetic-0", "synthetic-1", "synthetic-2"]
        preprocessing = {"size": 32, "range": [0, 1], "normalization": None}
    else:
        # PIL -> augmentation -> float [0,1] -> noise -> normalization.
        if weights is not None:
            mean, std = weights.transforms().mean, weights.transforms().std
            val_transform = weights.transforms()
        else:
            mean, std = [0.5] * 3, [0.5] * 3
            val_transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), ImpulseNoise(noise), transforms.Normalize(mean, std)])
        train_data = datasets.ImageFolder(data / "train", transform=train_transform)
        val_data = datasets.ImageFolder(data / "val", transform=val_transform)
        if train_data.class_to_idx != val_data.class_to_idx:
            raise ValueError("train/ and val/ must contain the same class folders")
        classes = train_data.classes
        preprocessing = {"size": 224, "validation": "resize 256 + center crop 224",
                         "mean": mean, "std": std, "range_before_normalization": [0, 1]}
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)
    model = resnet18(weights=weights)
    model.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    history = []
    for epoch in range(epochs):
        # Frozen backbone includes BatchNorm buffers, not only parameter gradients.
        model.eval()
        model.fc.train()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                correct += (model(images).argmax(1) == labels).sum().item()
        history.append({"epoch": epoch + 1, "validation_accuracy": correct / len(val_data)})
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": "own-image-data" if data is not None else "synthetic mechanism demo",
        "backbone_weights": weights.name if weights is not None else "random",
        "classes": classes, "preprocessing": preprocessing,
        "noise_amount": noise, "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "frozen_parameters_unchanged": all(torch.equal(before[name], p) for name, p in model.named_parameters() if not p.requires_grad),
        "frozen_buffers_unchanged": all(torch.equal(before[name], b) for name, b in model.named_buffers()),
        "head_changed": any(not torch.equal(before[name], p) for name, p in model.named_parameters() if name.startswith("fc.")),
        "history": history,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": report}, output / "head-model.pt")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['mode']} / {report['backbone_weights']} backbone; "
          f"{report['trainable_parameters']} trainable parameters")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/vision-head"))
    parser.add_argument("--data", type=Path, help="Root containing train/class/ and val/class/")
    parser.add_argument("--pretrained", action="store_true", help="May download ImageNet weights")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.02)
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.noise <= 1:
        parser.error("Use positive --epochs and --noise in [0,1]")
    if args.pretrained and args.data is None:
        parser.error("--pretrained needs --data; offline demo intentionally uses random weights")
    run(args.output, args.data, args.pretrained, args.epochs, args.noise)
