"""Train a reproducible EMNIST letter classifier with public TorchVision data.

The default run is deliberately small: it downloads EMNIST Letters when
needed, trains on a fixed subset, and saves figures under artifacts/. Device
selection is automatic; pass --device cpu to require CPU.
Use --full --device cuda for a longer run on a compatible GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import time
from pathlib import Path

try:
    from .validation_patterns import split_indices, BestState
except ImportError:  # direct script execution
    from validation_patterns import split_indices, BestState

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
LETTERS = list(string.ascii_uppercase)


class DenseLetterClassifier(nn.Module):
    def __init__(self, classes: int = 26) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, 256), nn.ReLU(), nn.Linear(256, classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ConvolutionalLetterClassifier(nn.Module):
    def __init__(self, classes: int = 26) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or --device auto.")
    return device


def deterministic_subset(dataset, limit: int | None, seed: int) -> Subset | datasets.EMNIST:
    if limit is None or limit >= len(dataset):
        return dataset
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:limit].tolist()
    return Subset(dataset, indices)


def load_emnist(data_root: Path, *, train: bool, transform, target_transform):
    """Load EMNIST and retry a Windows cleanup race after a completed download."""
    options = {
        "split": "letters",
        "train": train,
        "download": True,
        "transform": transform,
        "target_transform": target_transform,
    }
    try:
        return datasets.EMNIST(data_root, **options)
    except PermissionError:
        # TorchVision has already extracted the files in this case; a short retry
        # lets a transient OneDrive/antivirus file handle clear on Windows.
        time.sleep(1)
        return datasets.EMNIST(data_root, **options)


def train_epoch(model, loader, optimizer, loss_fn, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = total_correct = total_count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
    return total_loss / total_count, total_correct / total_count


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device: torch.device) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = total_correct = total_count = 0
    labels_all, predictions_all = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        total_loss += loss_fn(logits, labels.to(device)).item() * labels.size(0)
        predictions = logits.argmax(1).cpu()
        total_correct += (predictions == labels).sum().item()
        total_count += labels.size(0)
        labels_all.append(labels)
        predictions_all.append(predictions)
    return total_loss / total_count, total_correct / total_count, torch.cat(labels_all), torch.cat(predictions_all)


def save_figures(model, loader, labels, predictions, history, output: Path, device: torch.device) -> None:
    output.mkdir(parents=True, exist_ok=True)
    images, actual = next(iter(loader))
    with torch.inference_mode():
        predicted = model(images.to(device)).argmax(1).cpu()
    figure, axes = plt.subplots(3, 4, figsize=(8, 6))
    for axis, image, truth, guess in zip(axes.flat, images[:12], actual[:12], predicted[:12]):
        axis.imshow(image.squeeze(0).mul(0.5).add(0.5), cmap="gray")
        axis.set_title(f"{LETTERS[truth]} → {LETTERS[guess]}", color="#1b7f3a" if truth == guess else "#b23a48")
        axis.axis("off")
    figure.suptitle("EMNIST predictions from this run")
    figure.tight_layout()
    figure.savefig(output / "emnist-predictions.png", dpi=160)
    plt.close(figure)

    confusion = torch.zeros(26, 26, dtype=torch.int64)
    for truth, guess in zip(labels, predictions):
        confusion[truth, guess] += 1
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set(title="EMNIST confusion matrix from this run", xlabel="Predicted letter", ylabel="True letter")
    axis.set_xticks(range(26), LETTERS, fontsize=7)
    axis.set_yticks(range(26), LETTERS, fontsize=7)
    figure.colorbar(image, ax=axis, label="count")
    figure.tight_layout()
    figure.savefig(output / "emnist-confusion-matrix.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["validation_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="epoch"); axes[0].legend()
    axes[1].plot(history["train_accuracy"], label="train")
    axes[1].plot(history["validation_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="epoch"); axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "emnist-training-curves.png", dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = device_for(args.device)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ])
    target = lambda value: value - 1  # EMNIST Letters labels are 1–26; CrossEntropyLoss expects 0–25.
    data_root = Path(args.data_dir)
    train_data = load_emnist(data_root, train=True, transform=transform, target_transform=target)
    test_data = load_emnist(data_root, train=False, transform=transform, target_transform=target)
    train_limit = None if args.full else args.train_limit
    test_limit = None if args.full else args.test_limit
    train_data = deterministic_subset(train_data, train_limit, args.seed)
    test_data = deterministic_subset(test_data, test_limit, args.seed + 1)
    train_indices, val_indices = split_indices(len(train_data), args.validation_fraction, args.seed)
    # The deterministic preprocessing can be shared; only the indices differ.
    val_data = Subset(train_data, val_indices)
    source_indices = train_data.indices if isinstance(train_data, Subset) else range(len(train_data))
    split_record = {"training_source_indices": [source_indices[i] for i in train_indices],
                    "validation_source_indices": [source_indices[i] for i in val_indices]}
    train_data = Subset(train_data, train_indices)
    loader_args = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    test_loader = DataLoader(test_data, shuffle=False, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    model = ConvolutionalLetterClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    epochs = args.full_epochs if args.full else args.epochs
    history = {key: [] for key in ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy")}
    best = BestState()
    for epoch_index in range(epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss); history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(val_loss); history["validation_accuracy"].append(val_accuracy)
        best.consider(model, val_loss, epoch_index + 1)
    best.restore(model)
    test_loss, test_accuracy, labels, predictions = evaluate(model, test_loader, loss_fn, device)
    output = Path(args.output_dir)
    save_figures(model, test_loader, labels, predictions, history, output, device)
    torch.save({"model_state": model.state_dict(), "classes": LETTERS, "history": history, "best_epoch": best.epoch, "normalization": {"mean": [0.5], "std": [0.5]}, "split": split_record}, output / "emnist-model.pt")
    summary = {"device": str(device), "train_examples": len(train_data), "test_examples": len(test_data), "epochs": epochs, "validation_examples": len(val_data), "best_epoch": best.epoch, "selection": "minimum validation loss", "split": split_record, "test_loss": test_loss, "test_accuracy": test_accuracy}
    (output / "emnist-metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def smoke_test() -> dict[str, int]:
    torch.manual_seed(42)
    batch = torch.randn(8, 1, 28, 28)
    dense, cnn = DenseLetterClassifier(), ConvolutionalLetterClassifier()
    assert dense(batch).shape == cnn(batch).shape == (8, 26)
    return {"dense_parameters": sum(p.numel() for p in dense.parameters()), "cnn_parameters": sum(p.numel() for p in cnn.parameters())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=ROOT / "data" / "emnist", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "artifacts" / "emnist", type=Path)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--full-epochs", default=12, type=int)
    parser.add_argument("--train-limit", default=4000, type=int)
    parser.add_argument("--test-limit", default=1000, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--full", action="store_true", help="Use the complete dataset and the longer epoch budget.")
    parser.add_argument("--smoke-test", action="store_true", help="Check shapes without downloading data.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(smoke_test() if arguments.smoke_test else run(arguments), indent=2))
