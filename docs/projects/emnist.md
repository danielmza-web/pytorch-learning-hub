---
title: EMNIST letter classifier
course: PyTorch Fundamentals
tags:
  - project
  - emnist
  - cnn
last_reviewed: 2026-08-05
---

# Project: EMNIST letter classifier

## Problem

Classify 28 × 28 grayscale handwritten letters and compare a dense baseline with a convolutional model.

## Data contract

```text
input:  [batch, 1, 28, 28]
output: [batch, 26]
labels: integer class ids 0–25
```

Some EMNIST splits expose letter labels as `1–26`; the pipeline must remap them to the zero-based class indices expected by `CrossEntropyLoss`.

## Baseline

```python
nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 26),
)
```

Flattening discards explicit two-dimensional structure.

## CNN

```text
1×28×28 → Conv(32) → Pool → 32×14×14
          → Conv(64) → Pool → 64×7×7
          → Adaptive pool → 64
          → Linear → 26 logits
```

The CNN can reuse local detectors for strokes, curves, corners, and loops.

## Evaluation design

- Overall test accuracy.
- Per-letter recall.
- Confusion matrix for visually similar pairs.
- A separately reported external-handwriting sample.

An improvement on the EMNIST test split does not guarantee improvement on one person's handwriting. That external gap is a useful example of [distribution shift](../concepts/generalization.md#distribution-shift).

## Reproducibility

`examples/emnist_model.py` defines both architectures and verifies their input/output shapes without downloading data. A full dataset training run is intentionally separate from the fast documentation test.

