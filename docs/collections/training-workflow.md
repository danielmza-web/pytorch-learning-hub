---
title: Build a training workflow
tags:
  - collection
  - training
last_reviewed: 2026-08-06
---

# Build a training workflow

This collection connects data loading, model definition, optimization, and evaluation into one repeatable system. It is the useful bridge between a tensor experiment and a real image classifier.

## The contract between parts

| Part | Owns | Produces |
| --- | --- | --- |
| `Dataset` | one valid example | `(input, label)` |
| `DataLoader` | batching and order | batches of examples |
| model | a transformation | logits shaped `[batch, classes]` |
| loss | training objective | one differentiable scalar |
| optimizer | parameter updates | changed model weights |
| evaluation | held-out evidence | metrics and error patterns |

The boundaries matter. A Dataset should describe one sample. A DataLoader should not invent labels. A model should not silently apply a training-only transform. Evaluation should not update parameters.

## Questions to open

| If you are asking… | Open this |
| --- | --- |
| How do `Dataset`, `DataLoader`, and transforms divide responsibility? | [Data pipeline](../courses/fundamentals/data-pipeline.md) |
| Why use logits with `CrossEntropyLoss`? | [Models and training](../courses/fundamentals/models-training.md) |
| Why use both `model.eval()` and `torch.no_grad()`? | [Evaluation](../courses/fundamentals/evaluation.md) |
| How does the full loop work with real letters? | [EMNIST project](../projects/emnist.md) |

## End-to-end reference project

The EMNIST project downloads the public Letters split through TorchVision, remaps labels from `1–26` to `0–25`, trains a CNN, then saves predictions, curves, a confusion matrix, a checkpoint, and JSON metrics. Its default run limits data and epochs for a reproducible CPU result; `--full --device cuda` raises the budget without changing the workflow.

## Keep nearby

- Use random augmentation only on training data; validation should remain deterministic.
- Record the split seed, preprocessing, class names, epoch budget, and checkpoint with every result.
- Accuracy is one summary. Predictions and a confusion matrix reveal where it fails.
