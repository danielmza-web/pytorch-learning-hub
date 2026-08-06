"""Generate the conceptual PyTorch visuals used by the documentation.

These figures explain shapes and operations. They are deliberately not model
benchmarks or retained experiment output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "images"
NAVY = "#09233d"
BLUE = "#2ca7db"
ORANGE = "#eb7f30"
INK = "#16283a"
MUTED = "#587086"


def save(figure: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def box(axis, xy, width, height, title, subtitle, color=BLUE):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.04",
                           linewidth=1.8, edgecolor=color, facecolor="#f7fbfd")
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height * .62, title, ha="center", va="center",
              fontsize=11, weight="bold", color=NAVY)
    axis.text(xy[0] + width / 2, xy[1] + height * .32, subtitle, ha="center", va="center",
              fontsize=8.5, color=MUTED)


def arrow(axis, start, end, label=""):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                   linewidth=1.5, color=ORANGE))
    if label:
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .11, label,
                  ha="center", va="bottom", fontsize=8, color=MUTED)


def data_pipeline() -> None:
    figure, axis = plt.subplots(figsize=(11, 3.6))
    axis.set(xlim=(0, 12), ylim=(0, 4.1)); axis.axis("off")
    box(axis, (.25, 1.25), 2.05, 1.55, "Image file", "H × W × C pixels")
    for x in np.linspace(.48, 1.7, 5):
        for y in np.linspace(1.55, 2.25, 4):
            axis.add_patch(Rectangle((x, y), .2, .16, facecolor=BLUE, alpha=.2 + (x + y) % .4, edgecolor="none"))
    box(axis, (3.25, 1.25), 2.05, 1.55, "Transform", "resize · tensor · normalize")
    box(axis, (6.25, 1.25), 2.05, 1.55, "Dataset", "one tensor + one label")
    box(axis, (9.25, 1.25), 2.35, 1.55, "DataLoader", "batch: [B, C, H, W]", ORANGE)
    arrow(axis, (2.35, 2.02), (3.18, 2.02), "prepare")
    arrow(axis, (5.35, 2.02), (6.18, 2.02), "return")
    arrow(axis, (8.35, 2.02), (9.18, 2.02), "batch")
    axis.text(6, .38, "Illustrative data path: the model sees the final batch tensor, not the original file.",
              ha="center", fontsize=9, color=INK)
    save(figure, "data-pipeline-visual.png")


def convolution() -> None:
    values = np.array([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, .1, .9, 1, 1],
                       [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    response = np.array([[0, 1, 3, 2], [0.1, 1.8, 2.8, 2], [0.1, 1.8, 2.8, 2], [0, 1, 3, 2]])
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), gridspec_kw={"width_ratios": [1.35, .9, 1]})
    for axis, matrix, title, cmap in zip(axes, (values, kernel, response),
                                          ("Input image", "3 × 3 edge kernel", "Feature map"),
                                          ("Blues", "coolwarm", "Oranges")):
        image = axis.imshow(matrix, cmap=cmap, vmin=-1 if title.startswith("3") else None,
                            vmax=1 if title.startswith("3") else None)
        axis.set_title(title, color=NAVY, weight="bold", fontsize=11)
        axis.set_xticks(np.arange(-.5, matrix.shape[1], 1), minor=True)
        axis.set_yticks(np.arange(-.5, matrix.shape[0], 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.4)
        axis.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[0].add_patch(Rectangle((1.5, 1.5), 3, 3, fill=False, linewidth=2.4, edgecolor=ORANGE))
    figure.text(.5, .03, "Illustrative: one hand-designed vertical-edge filter slides across an image. A CNN learns its filters from data.",
                ha="center", fontsize=9, color=INK)
    save(figure, "convolution-visual.png")


def cnn_shape_path() -> None:
    figure, axis = plt.subplots(figsize=(11, 3.6))
    axis.set(xlim=(0, 14), ylim=(0, 4.3)); axis.axis("off")
    items = [
        (.3, "Image", "[B, 3, 32, 32]", BLUE),
        (3.0, "Conv + ReLU", "[B, 32, 32, 32]", BLUE),
        (5.7, "MaxPool", "[B, 32, 16, 16]", ORANGE),
        (8.4, "Conv + ReLU", "[B, 64, 16, 16]", BLUE),
        (11.1, "Pool + classifier", "[B, classes]", ORANGE),
    ]
    for index, (x, title, shape, color) in enumerate(items):
        box(axis, (x, 1.3), 2.1, 1.55, title, shape, color)
        if index:
            arrow(axis, (x - .55, 2.07), (x - .04, 2.07))
    axis.text(7, .38, "Channels grow as the model learns more feature maps; pooling reduces spatial detail before classification.",
              ha="center", fontsize=9, color=INK)
    save(figure, "cnn-shape-visual.png")


def training_cycle() -> None:
    figure, axis = plt.subplots(figsize=(10.5, 4.1))
    axis.set(xlim=(0, 12), ylim=(0, 5)); axis.axis("off")
    items = [
        (1.1, 3.35, "Batch", "images + labels", BLUE),
        (4.0, 3.35, "Model", "logits", BLUE),
        (6.9, 3.35, "Loss", "one error value", ORANGE),
        (6.9, .85, "backward()", "gradients", ORANGE),
        (4.0, .85, "step()", "updated weights", BLUE),
    ]
    for x, y, title, subtitle, color in items:
        box(axis, (x, y), 2.05, 1.05, title, subtitle, color)
    arrow(axis, (3.2, 3.87), (3.92, 3.87)); arrow(axis, (6.1, 3.87), (6.82, 3.87))
    arrow(axis, (7.92, 3.25), (7.92, 1.98), "loss.backward()")
    arrow(axis, (6.82, 1.37), (6.1, 1.37)); arrow(axis, (4.0, 1.37), (2.05, 3.25), "next batch")
    axis.text(6, .18, "Illustrative update cycle: only training performs backpropagation and changes parameters.",
              ha="center", fontsize=9, color=INK)
    save(figure, "training-cycle-visual.png")


if __name__ == "__main__":
    data_pipeline()
    convolution()
    cnn_shape_path()
    training_cycle()
