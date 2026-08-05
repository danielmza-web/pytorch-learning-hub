---
title: Project gallery
tags:
  - projects
last_reviewed: 2026-08-05
---

# Original project gallery

These case studies turn a PyTorch idea into a practical decision. Each page starts short, explains the important code, distinguishes evidence from a conceptual visual, and ends with the complete runnable source.

| Project | Question to revisit | Visual evidence | Complete source |
| --- | --- | --- | --- |
| [Nonlinear regression](regression.md) | When is a line not enough? | Reproduced prediction chart | Fast deterministic training script |
| [EMNIST letters](emnist.md) | Why preserve image structure? | Retained CPU-run predictions, curves, and confusion matrix | Downloads EMNIST and trains a CNN |
| [Robust image pipeline](robust-image-pipeline.md) | How do bad files affect training? | Verified validation decision flow | Downloads CIFAR-10, validates files, and trains a CNN |
| [Nature CNN](nature-cnn.md) | How do we reason about overfitting? | Architecture map + illustrative curve | Downloads CIFAR-100 and trains a nature-class CNN |

!!! note "Results policy"
    Only results produced by retained, reproducible runs are reported as measurements. Conceptual curves are labelled illustrative, and smoke tests are not presented as model-quality benchmarks.
