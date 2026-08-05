"""Build, validate, and train on a real folder-based image pipeline.

The project downloads CIFAR-10 through TorchVision, materializes a small
five-class image-folder dataset, deliberately adds one corrupt file to prove
the validation path, then trains a compact CNN and writes artifacts. Default
settings are small enough for CPU; use --full --device cuda for longer runs.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("airplane", "automobile", "bird", "cat", "dog")


@dataclass(frozen=True)
class InvalidImage:
    path: Path
    reason: str


class ValidatedImageFolder(Dataset):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root: Path, transform=None) -> None:
        self.root, self.transform = Path(root), transform
        self.class_names = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.samples: list[tuple[Path, int]] = []; self.invalid: list[InvalidImage] = []
        for class_name in self.class_names:
            for path in sorted((self.root / class_name).iterdir()):
                if path.suffix.lower() not in self.extensions: continue
                try:
                    with Image.open(path) as image: image.verify()
                except Exception as error:
                    self.invalid.append(InvalidImage(path, type(error).__name__)); continue
                self.samples.append((path, self.class_to_index[class_name]))

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image: sample = image.convert("RGB")
        return (self.transform(sample) if self.transform else sample), label


class FolderCNN(nn.Module):
    def __init__(self, classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.network(x)


def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable; use --device cpu or --device auto.")
    return torch.device(name)


def materialize_cifar10(data_root: Path, image_root: Path, per_class: int, seed: int) -> None:
    """Download official CIFAR-10 and make a deterministic, human-readable folder dataset."""
    source = datasets.CIFAR10(data_root, train=True, download=True)
    image_root.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in CLASSES}; class_ids = {name: source.classes.index(name) for name in CLASSES}
    order = torch.randperm(len(source), generator=torch.Generator().manual_seed(seed)).tolist()
    for index in order:
        image, label = source[index]; name = source.classes[label]
        if name not in class_ids or counts[name] >= per_class: continue
        folder = image_root / name; folder.mkdir(exist_ok=True)
        target = folder / f"{counts[name]:04d}.png"
        if not target.exists(): image.save(target)
        counts[name] += 1
        if all(value >= per_class for value in counts.values()): break
    if any(value < per_class for value in counts.values()): raise RuntimeError(f"Could not materialize every requested class: {counts}")
    broken = image_root / CLASSES[0] / "corrupt-example.jpg"
    if not broken.exists(): broken.write_bytes(b"intentionally not an image")


def split_indices(total: int, validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist(); cut = int(total * (1 - validation_fraction))
    return order[:cut], order[cut:]


def train_epoch(model, loader, optimizer, loss_fn, device, train: bool) -> tuple[float, float]:
    model.train(train); loss_total = correct = count = 0; context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train: optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = loss_fn(logits, labels)
            if train: loss.backward(); optimizer.step()
            loss_total += loss.item() * labels.size(0); correct += (logits.argmax(1) == labels).sum().item(); count += labels.size(0)
    return loss_total / count, correct / count


@torch.inference_mode()
def collect(model, loader, device):
    model.eval(); images_all = []; labels_all = []; predictions_all = []
    for images, labels in loader:
        images_all.append(images); labels_all.append(labels); predictions_all.append(model(images.to(device)).argmax(1).cpu())
    return torch.cat(images_all), torch.cat(labels_all), torch.cat(predictions_all)


def save_artifacts(images, labels, predictions, history, dataset: ValidatedImageFolder, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 4, figsize=(9, 7))
    for axis, image, truth, guess in zip(axes.flat, images[:12], labels[:12], predictions[:12]):
        axis.imshow(image.permute(1, 2, 0).clamp(0, 1)); axis.set_title(f"{dataset.class_names[truth]} → {dataset.class_names[guess]}", fontsize=8, color="#1b7f3a" if truth == guess else "#b23a48"); axis.axis("off")
    figure.suptitle("Validated-folder predictions from this run"); figure.tight_layout(); figure.savefig(output / "pipeline-predictions.png", dpi=160); plt.close(figure)
    matrix = torch.zeros(len(dataset.class_names), len(dataset.class_names), dtype=torch.int64)
    for truth, guess in zip(labels, predictions): matrix[truth, guess] += 1
    figure, axis = plt.subplots(figsize=(7, 6)); chart = axis.imshow(matrix, cmap="Blues")
    axis.set(title="Validated-folder confusion matrix", xlabel="Predicted", ylabel="True"); axis.set_xticks(range(len(dataset.class_names)), dataset.class_names, rotation=35, ha="right"); axis.set_yticks(range(len(dataset.class_names)), dataset.class_names); figure.colorbar(chart, ax=axis); figure.tight_layout(); figure.savefig(output / "pipeline-confusion-matrix.png", dpi=160); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["validation_loss"], label="validation"); axes[0].set(title="Loss", xlabel="epoch"); axes[0].legend()
    axes[1].plot(history["train_accuracy"], label="train"); axes[1].plot(history["validation_accuracy"], label="validation"); axes[1].set(title="Accuracy", xlabel="epoch"); axes[1].legend()
    figure.tight_layout(); figure.savefig(output / "pipeline-training-curves.png", dpi=160); plt.close(figure)
    report = {"valid_samples": len(dataset), "invalid_samples": [{"path": str(item.path), "reason": item.reason} for item in dataset.invalid], "classes": dataset.class_names}
    (output / "pipeline-validation-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed); device = choose_device(args.device); root = Path(args.image_root)
    materialize_cifar10(Path(args.data_dir), root, args.full_per_class if args.full else args.per_class, args.seed)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ValidatedImageFolder(root, transform=transform)
    train_indices, validation_indices = split_indices(len(dataset), args.validation_fraction, args.seed)
    loader = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(dataset, train_indices), shuffle=True, **loader); validation_loader = DataLoader(Subset(dataset, validation_indices), shuffle=False, **loader)
    model = FolderCNN(len(dataset.class_names)).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4); loss_fn = nn.CrossEntropyLoss()
    epochs = args.full_epochs if args.full else args.epochs; history = {key: [] for key in ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy")}
    for _ in range(epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, loss_fn, device, True); validation_loss, validation_accuracy = train_epoch(model, validation_loader, optimizer, loss_fn, device, False)
        history["train_loss"].append(train_loss); history["train_accuracy"].append(train_accuracy); history["validation_loss"].append(validation_loss); history["validation_accuracy"].append(validation_accuracy)
    images, labels, predictions = collect(model, validation_loader, device); output = Path(args.output_dir); save_artifacts(images, labels, predictions, history, dataset, output)
    torch.save({"model_state": model.state_dict(), "classes": dataset.class_names, "history": history}, output / "pipeline-model.pt")
    summary = {"device": str(device), "valid_samples": len(dataset), "invalid_samples": len(dataset.invalid), "epochs": epochs, "validation_accuracy": history["validation_accuracy"][-1]}
    (output / "pipeline-metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def smoke_test() -> dict[str, int]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((16, 16))])
    with TemporaryDirectory() as temporary:
        root = Path(temporary); (root / "bee").mkdir(); (root / "rose").mkdir()
        Image.new("RGB", (12, 10), "gold").save(root / "bee" / "valid.png"); Image.new("RGB", (9, 14), "crimson").save(root / "rose" / "valid.png"); (root / "bee" / "broken.jpg").write_bytes(b"not an image")
        dataset = ValidatedImageFolder(root, transform=transform); sample, label = dataset[0]
        assert dataset.class_names == ["bee", "rose"] and len(dataset) == 2 and len(dataset.invalid) == 1 and sample.shape == (3, 16, 16) and label in (0, 1)
        return {"valid_samples": len(dataset), "invalid_samples": len(dataset.invalid)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=ROOT / "data" / "cifar10", type=Path); parser.add_argument("--image-root", default=ROOT / "data" / "validated-cifar10", type=Path); parser.add_argument("--output-dir", default=ROOT / "artifacts" / "robust-pipeline", type=Path)
    parser.add_argument("--epochs", default=5, type=int); parser.add_argument("--full-epochs", default=25, type=int); parser.add_argument("--per-class", default=80, type=int); parser.add_argument("--full-per-class", default=1000, type=int); parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--validation-fraction", default=0.2, type=float); parser.add_argument("--learning-rate", default=1e-3, type=float); parser.add_argument("--seed", default=42, type=int); parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); parser.add_argument("--full", action="store_true"); parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args(); print(json.dumps(smoke_test() if arguments.smoke_test else run(arguments), indent=2))
