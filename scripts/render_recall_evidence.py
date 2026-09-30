"""Render original recall figures from retained reports; never replace legacy results."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from recall_patterns import ImpulseNoise


def render():
    plt.rcParams.update({"font.size": 12})
    reports = ROOT / "docs/assets/data/recall-2026-10-01"
    images = ROOT / "docs/assets/images"
    training = json.loads((reports / "comparison.json").read_text())
    for key, title, ylabel, filename in [
        ("loss", "Validation loss · 48 synthetic samples", "Cross-entropy", "training-loss.png"),
        ("macro_f1", "Validation macro F1 · 3 classes", "Macro F1", "training-f1.png")]:
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        for trial in training["trials"]:
            ax.plot([h["epoch"] for h in trial["history"]], [h[key] for h in trial["history"]], marker="o", label=f"Initial LR {trial['initial_lr']}")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(images / filename, dpi=160); plt.close(fig)
    best = next(t for t in training["trials"] if t["initial_lr"] == training["best_initial_lr"])
    matrix = np.array(best["history"][-1]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    chart = ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center", color="white" if value > matrix.max()/2 else "black", fontsize=15)
    ax.set(title=f"Final validation · initial LR {best['initial_lr']}", xlabel="Predicted class", ylabel="True class", xticks=range(3), yticks=range(3))
    fig.colorbar(chart, ax=ax, label="Samples"); fig.tight_layout()
    fig.savefig(images / "training-confusion.png", dpi=160); plt.close(fig)

    losses = [1, .8, .8, .8, .7, .7, .7, .7, .7, .7, .7, .7]
    fig, ax = plt.subplots(figsize=(6, 4))
    schedules = {}
    for name in ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]:
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.5) if name == "StepLR" else
                     torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12) if name == "CosineAnnealingLR" else
                     torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=1))
        rates = []
        for loss in losses:
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step(loss) if name == "ReduceLROnPlateau" else scheduler.step()
        schedules[name] = rates
        ax.plot(range(1, 13), rates, marker="o", label=name)
    ax.set(title="Toy schedule calls · initial LR 0.1", xlabel="Epoch (LR used before scheduler call)", ylabel="Learning rate")
    ax.legend(fontsize=10); ax.grid(alpha=.2); fig.tight_layout()
    svg = images / "training-schedules.svg"
    fig.savefig(svg); plt.close(fig)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    (reports / "schedules.json").write_text(json.dumps({"kind": "toy scheduler execution, not training", "validation_loss_signal": losses, "rates_used": schedules}, indent=2), encoding="utf-8")

    # An original inspection scene, not a photograph or a course image.
    y, x = np.mgrid[:180, :220]
    pixels = np.stack([70 + x*.3, 100 + y*.3, 150 + x*.1], axis=-1).astype(np.uint8)
    pixels[25:155, 25:195] = [175, 191, 205]
    pixels[65:115, 145:150] = [55, 70, 85]
    pixels[(x-75)**2 + (y-75)**2 < 140] = [75, 90, 100]
    original = Image.fromarray(pixels)
    torch.manual_seed(71)
    crop = transforms.RandomResizedCrop((180, 220), scale=(.65, .85))(original)
    colour = transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3)(crop)
    noisy = ImpulseNoise(.05)(transforms.ToTensor()(colour))
    panels = [np.asarray(original), np.asarray(crop), np.asarray(colour), noisy.permute(1, 2, 0).numpy()]
    # 2x2 layout stays legible when scaled to a mobile column.
    fig, axes = plt.subplots(2, 2, figsize=(6, 5.5))
    for ax, panel, title in zip(axes.flat, panels, ["Original illustration", "RandomResizedCrop", "ColorJitter after crop", "ImpulseNoise (p = 0.05)"]):
        ax.imshow(panel); ax.set_title(title, fontsize=15); ax.axis("off")
    fig.tight_layout(); fig.savefig(images / "vision-transforms.png", dpi=160); plt.close(fig)
    normalized = transforms.Normalize([.5]*3, [.5]*3)(noisy)
    report = {"kind": "seeded execution on an original illustration", "seed": 71,
              "transforms": ["RandomResizedCrop: scale .65–.85, output 180×220", "ColorJitter: brightness/contrast/saturation .3", "ToTensor", "ImpulseNoise .05", "Normalize mean=.5 std=.5"],
              "float_shape": list(noisy.shape), "float_range": [noisy.min().item(), noisy.max().item()],
              "normalized_range": [normalized.min().item(), normalized.max().item()]}
    (reports / "transforms.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Rendered training loss, F1, confusion matrix and seeded transform panels from retained evidence")


if __name__ == "__main__":
    render()
