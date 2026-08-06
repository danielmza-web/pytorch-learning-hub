---
title: PyTorch learning hub
tags:
  - home
last_reviewed: 2026-08-06
---

# PyTorch learning hub

Learn or revisit PyTorch through connected explanations, visual tools, and complete runnable projects. The library is organized by guides, not by progress or course status.

[Start PyTorch Fundamentals](guides/fundamentals/core-workflow.md){ .md-button .md-button--primary }
[Open the project gallery](projects/index.md){ .md-button }

## Start here

PyTorch projects become easier to read when every part has one responsibility:

![Conceptual diagram showing a batch moving through model, logits, loss, gradients, and an optimizer update](assets/images/training-cycle-visual.png)

```text
data → batch → model → logits → loss → gradients → parameter update
                            ↓
                   validation and error inspection
```

Begin with the two long Fundamentals guides:

1. [Core workflow](guides/fundamentals/core-workflow.md) explains tensors, Dataset, DataLoader, models, logits, loss, autograd, optimization, and evaluation.
2. [Vision and real data](guides/fundamentals/vision-real-data.md) explains convolution, CNN shapes, robust image pipelines, generalization, regularization, and saving.

Each guide starts with a map and uses a table of contents for direct return visits. There are no parallel collections containing the same topic.

## Choose a question

| If you want to understand… | Open this section |
| --- | --- |
| What `[32, 3, 224, 224]` means | [Tensors and shapes](guides/fundamentals/core-workflow.md#tensors-shapes-dtype-and-device) |
| How files become shuffled batches | [Dataset, transforms, and DataLoader](guides/fundamentals/core-workflow.md#dataset-transforms-and-dataloader) |
| Why models return logits | [Models, activations, and logits](guides/fundamentals/core-workflow.md#models-activations-and-logits) |
| Where weights actually change | [Loss, autograd, and optimizers](guides/fundamentals/core-workflow.md#loss-autograd-and-optimizers) |
| Why a CNN preserves image structure | [Convolution and feature maps](guides/fundamentals/vision-real-data.md#convolution-and-feature-maps) |
| Why training improves while validation worsens | [Generalization and regularization](guides/fundamentals/vision-real-data.md#generalization-and-regularization) |
| Why a run fails | [Reference and troubleshooting](reference/index.md#common-errors) |

## Learn from complete examples

The four projects are maintained programs, not isolated fragments. Each page explains the question, important code, evidence, an interactive check, how to run a small CPU version, the optional longer GPU mode, and the complete source.

| Project | What it makes visible |
| --- | --- |
| [Nonlinear regression](projects/regression.md) | why an activation changes what a network can represent |
| [EMNIST letter classifier](projects/emnist.md) | a complete image-classification workflow and CNN shape path |
| [Robust image pipeline](projects/robust-image-pipeline.md) | validation, corrupt-file decisions, batches, and diagnostics |
| [Nature CNN](projects/nature-cnn.md) | reusable CNN blocks, regularization, curves, and saved artifacts |

## How this library grows

Every future PyTorch guide will use two substantial pages: one for its core mechanisms and one for applied workflows. A guide appears only when it contains useful material. Shared projects and the reference remain available without duplicating explanations.

For a short review outside the documentation site, open [DaZu's PyTorch guides](https://dazu.xyz/learn/pytorch/).
