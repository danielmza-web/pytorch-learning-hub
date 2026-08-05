---
title: Nature CNN
course: PyTorch Fundamentals
tags:
  - project
  - cifar100
  - overfitting
last_reviewed: 2026-08-05
---

# Project: nature CNN and overfitting

## Problem

Classify a curated CIFAR-100 subset containing flowers, mammals, and insects, then diagnose the gap between training and validation performance.

## Baseline architecture

```text
3×32×32
→ Conv 32 → ReLU → Pool
→ Conv 64 → ReLU → Pool
→ Conv 128 → ReLU → Pool
→ Flatten 2048 → Linear 512 → Dropout → class logits
```

## Regularized comparison

- Training-only augmentation.
- Batch normalization inside reusable convolution blocks.
- Stronger but realistic dropout.
- AdamW weight decay.
- Best-validation checkpoint selection.
- Per-class evaluation rather than accuracy alone.

## Experiment rules

1. Keep the same class subset and split.
2. Compare curves over the same epoch budget.
3. Record parameter count and inference time as well as accuracy.
4. Inspect confusion between visually related classes.
5. Reject augmentations that do not preserve the semantic label.

## Reproducibility

`examples/nature_cnn.py` verifies tensor shapes and parameter counts for baseline and modular architectures. Full CIFAR-100 training is kept out of the documentation build so updates remain fast and do not trigger hidden downloads.

!!! important
    A shape smoke test proves that the model executes; it does not prove predictive quality. Measured accuracy belongs in a retained experiment artifact, not in documentation written from memory.

