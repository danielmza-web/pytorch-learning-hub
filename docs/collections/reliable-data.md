---
title: Reliable image data
tags:
  - collection
  - data-quality
last_reviewed: 2026-08-06
---

# Reliable image data

Better architecture cannot repair a dataset whose labels, files, transforms, or split rules are unreliable. This collection treats data work as model work.

## Before training, establish the data contract

1. Define stable class names and class-to-index mapping.
2. Validate file extension, readability, color mode, and expected input shape.
3. Record rejected files with a path and reason rather than silently replacing them.
4. Split examples with a fixed seed before applying random training augmentation.
5. Inspect class counts in every split.
6. Make validation preprocessing deterministic.

## Questions to open

| If you are asking… | Open this |
| --- | --- |
| How should an image folder become a Dataset? | [Robust pipelines guide](../courses/fundamentals/robust-pipelines.md) |
| What belongs in a transform pipeline? | [Data pipeline](../courses/fundamentals/data-pipeline.md) |
| How can a corrupt image be handled honestly? | [Robust image pipeline project](../projects/robust-image-pipeline.md) |
| How does bad input turn into misleading validation? | [Common errors](../reference/common-errors.md) |

## End-to-end reference project

The robust-pipeline project downloads public CIFAR-10 images through TorchVision, materializes a deterministic five-class folder dataset, and intentionally adds one corrupt file. The Dataset reports that file during indexing, trains a compact CNN only on valid images, and saves a validation report, prediction grid, curves, confusion matrix, checkpoint, and metrics.

## Keep nearby

- Folder order is not a class-id contract; sort class names deliberately.
- Do not let random crops or flips leak into validation.
- A successful batch proves only that one batch can run. Inspect the manifest before trusting a metric.
