"""Train a small, reproducible CIFAR-100 nature classifier with real images.

Default settings are intentionally small and work on CPU; automatic device
selection uses CUDA when available. The script downloads
CIFAR-100 through TorchVision, filters 15 nature classes, trains a CNN, and
saves a sample grid, predictions, curves, confusion matrix, checkpoint, and
JSON metrics. Use --full --device cuda for a longer run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["beaver", "bear", "bee", "beetle", "butterfly", "caterpillar", "cockroach", "dolphin", "fox", "leopard", "lion", "otter", "tiger", "whale", "wolf"]


class NatureSubset(Dataset):
    def __init__(self, dataset: datasets.CIFAR100, limit_per_class: int | None, seed: int) -> None:
        self.dataset = dataset
        source_ids = {name: dataset.classes.index(name) for name in CLASS_NAMES}
        found = {name: 0 for name in CLASS_NAMES}
        indices = list(range(len(dataset)))
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(indices), generator=generator).tolist()
        self.samples: list[tuple[int, int]] = []
        for source_index in order:
            source_label = dataset.targets[source_index]
            name = dataset.classes[source_label]
            if name not in source_ids or (limit_per_class is not None and found[name] >= limit_per_class):
                continue
            self.samples.append((source_index, CLASS_NAMES.index(name)))
            found[name] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        source_index, label = self.samples[index]
        image, _ = self.dataset[source_index]
        return image, label


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NatureCNN(nn.Module):
    def __init__(self, classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.features = nn.Sequential(CNNBlock(3, 32), CNNBlock(32, 64), CNNBlock(64, 128))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable; use --device cpu or --device auto.")
    return torch.device(name)


def epoch(model, loader, optimizer, loss_fn, device: torch.device, train: bool):
    model.train(train); loss_total = correct = count = 0
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train: optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = loss_fn(logits, labels)
            if train: loss.backward(); optimizer.step()
            loss_total += loss.item() * labels.size(0); correct += (logits.argmax(1) == labels).sum().item(); count += labels.size(0)
    return loss_total / count, correct / count


@torch.inference_mode()
def collect_predictions(model, loader, device: torch.device):
    model.eval(); images_all, labels_all, predictions_all = [], [], []
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(1).cpu()
        images_all.append(images); labels_all.append(labels); predictions_all.append(predictions)
    return torch.cat(images_all), torch.cat(labels_all), torch.cat(predictions_all)


def save_artifacts(images, labels, predictions, history, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor((0.5071, 0.4867, 0.4408)).view(3, 1, 1); std = torch.tensor((0.2675, 0.2565, 0.2761)).view(3, 1, 1)
    figure, axes = plt.subplots(3, 4, figsize=(9, 7))
    for axis, image, truth, guess in zip(axes.flat, images[:12], labels[:12], predictions[:12]):
        axis.imshow(image.mul(std).add(mean).permute(1, 2, 0).clamp(0, 1))
        axis.set_title(f"{CLASS_NAMES[truth]} → {CLASS_NAMES[guess]}", fontsize=8, color="#1b7f3a" if truth == guess else "#b23a48")
        axis.axis("off")
    figure.suptitle("CIFAR-100 nature predictions from this run"); figure.tight_layout()
    figure.savefig(output / "nature-predictions.png", dpi=160); plt.close(figure)

    matrix = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)
    for truth, guess in zip(labels, predictions): matrix[truth, guess] += 1
    figure, axis = plt.subplots(figsize=(10, 8)); chart = axis.imshow(matrix, cmap="Blues")
    axis.set(title="CIFAR-100 confusion matrix from this run", xlabel="Predicted", ylabel="True")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=65, ha="right", fontsize=7)
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=7)
    figure.colorbar(chart, ax=axis, label="count"); figure.tight_layout()
    figure.savefig(output / "nature-confusion-matrix.png", dpi=160); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["test_loss"], label="test"); axes[0].set(title="Loss", xlabel="epoch"); axes[0].legend()
    axes[1].plot(history["train_accuracy"], label="train"); axes[1].plot(history["test_accuracy"], label="test"); axes[1].set(title="Accuracy", xlabel="epoch"); axes[1].legend()
    figure.tight_layout(); figure.savefig(output / "nature-training-curves.png", dpi=160); plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed); device = choose_device(args.device)
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    root = Path(args.data_dir)
    train_source = datasets.CIFAR100(root, train=True, download=True, transform=train_transform)
    test_source = datasets.CIFAR100(root, train=False, download=True, transform=test_transform)
    train_data = NatureSubset(train_source, None if args.full else args.train_per_class, args.seed)
    test_data = NatureSubset(test_source, None if args.full else args.test_per_class, args.seed + 1)
    loader = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, **loader); test_loader = DataLoader(test_data, shuffle=False, **loader)
    model = NatureCNN().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay); loss_fn = nn.CrossEntropyLoss()
    epochs = args.full_epochs if args.full else args.epochs; history = {key: [] for key in ("train_loss", "train_accuracy", "test_loss", "test_accuracy")}
    for _ in range(epochs):
        train_loss, train_accuracy = epoch(model, train_loader, optimizer, loss_fn, device, True)
        test_loss, test_accuracy = epoch(model, test_loader, optimizer, loss_fn, device, False)
        history["train_loss"].append(train_loss); history["train_accuracy"].append(train_accuracy); history["test_loss"].append(test_loss); history["test_accuracy"].append(test_accuracy)
    images, labels, predictions = collect_predictions(model, test_loader, device); output = Path(args.output_dir)
    save_artifacts(images, labels, predictions, history, output)
    torch.save({"model_state": model.state_dict(), "classes": CLASS_NAMES, "history": history}, output / "nature-cnn.pt")
    summary = {"device": str(device), "train_examples": len(train_data), "test_examples": len(test_data), "epochs": epochs, "test_accuracy": history["test_accuracy"][-1]}
    (output / "nature-metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def smoke_test() -> dict[str, int]:
    torch.manual_seed(42); batch = torch.randn(4, 3, 32, 32); model = NatureCNN()
    assert model(batch).shape == (4, len(CLASS_NAMES))
    return {"parameters": sum(parameter.numel() for parameter in model.parameters()), "classes": len(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=ROOT / "data" / "cifar100", type=Path); parser.add_argument("--output-dir", default=ROOT / "artifacts" / "nature-cnn", type=Path)
    parser.add_argument("--epochs", default=4, type=int); parser.add_argument("--full-epochs", default=20, type=int)
    parser.add_argument("--train-per-class", default=120, type=int); parser.add_argument("--test-per-class", default=40, type=int); parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float); parser.add_argument("--weight-decay", default=1e-4, type=float); parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); parser.add_argument("--full", action="store_true"); parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args(); print(json.dumps(smoke_test() if arguments.smoke_test else run(arguments), indent=2))
