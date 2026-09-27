---
title: PyTorch learning hub
tags:
  - home
last_reviewed: 2026-09-27
---

<div class="hub-hero" markdown="1">

<span class="eyebrow">A practical PyTorch reference</span>

# Understand the step. Remember the reason.

Follow a model from its first tensor to a trustworthy result. Use the short explanations when learning and return to a specific question when coding.

Start with Course 1's working path. For each step, predict the output, run the small code example, and answer the check question. Notes on the Course 2 labs studied so far are in the reference.

[Start with the training path](guides/fundamentals/core-workflow.md){ .md-button .md-button--primary }
[Choose a runnable project](projects/index.md){ .md-button }

</div>

## Pick a path

<div class="topic-grid" markdown="1">

<div class="topic-card" markdown="1">

**I'm learning the workflow**

Start with shapes, batches, model output, loss, and the update step.

[Build and train](guides/fundamentals/core-workflow.md)

</div>

<div class="topic-card" markdown="1">

**I'm working with images**

See how convolution changes shapes and why data checks and validation matter.

[Images and CNNs](guides/fundamentals/vision-real-data.md)

</div>

<div class="topic-card" markdown="1">

**I need a working example**

Choose a small project by the question it helps answer.

[Choose a project](projects/index.md)

</div>

<div class="topic-card" markdown="1">

**Something is going wrong**

Use a short checklist to inspect shapes, labels, devices, and evaluation.

[Quick reference](reference/index.md)

</div>

</div>

## The workflow at a glance

One mental model connects the pages:

![Conceptual diagram showing a batch moving through model, logits, loss, gradients, and an optimizer update](assets/images/training-cycle-visual.png)

**Data → batch → model output → loss → gradients → update.** Classification models often output logits; regression models output values. Validation checks whether the learned pattern works beyond the training batch.

Begin with the two connected Fundamentals pages:

1. [Core workflow](guides/fundamentals/core-workflow.md) explains tensors, Dataset, DataLoader, models, logits, loss, autograd, optimization, and evaluation.
2. [Vision and real data](guides/fundamentals/vision-real-data.md) explains convolution, CNN shapes, robust image pipelines, generalization, regularization, and saving.

Each page starts with a map and has direct section links for return visits.

## Choose a question

| If you want to understand… | Open this section |
| --- | --- |
| What `[32, 3, 224, 224]` means | [Tensors and shapes](guides/fundamentals/core-workflow.md#tensors-shapes-dtype-and-device) |
| How files become shuffled batches | [Dataset, transforms, and DataLoader](guides/fundamentals/core-workflow.md#dataset-transforms-and-dataloader) |
| When a classifier returns logits | [Models, activations, and logits](guides/fundamentals/core-workflow.md#models-activations-and-logits) |
| Why ReLU makes a hidden layer nonlinear | [Why activations matter](guides/fundamentals/core-workflow.md#why-activations-matter) |
| Where weights actually change | [Loss, autograd, and optimizers](guides/fundamentals/core-workflow.md#loss-autograd-and-optimizers) |
| Why a CNN preserves image structure | [Convolution and feature maps](guides/fundamentals/vision-real-data.md#convolution-and-feature-maps) |
| Why training improves while validation worsens | [Generalization and regularization](guides/fundamentals/vision-real-data.md#generalization-and-regularization) |
| What to inspect when a run fails | [Reference and troubleshooting](reference/index.md#common-errors) |
| How metrics, schedules, and transfer learning fit Course 2 | [Course 2 notes](reference/index.md#course-2-what-changes-in-practice) |

## Learn from complete examples

The four projects are maintained programs, not isolated fragments. Each page connects one question to code, evidence, a small interactive check, and complete source. The image projects use limited data by default and choose CUDA when available; `--device cpu` forces a CPU run.

| Project | What it makes visible |
| --- | --- |
| [Nonlinear regression](projects/regression.md) | why an activation changes what a network can represent |
| [EMNIST letter classifier](projects/emnist.md) | a complete image-classification workflow and CNN shape path |
| [Robust image pipeline](projects/robust-image-pipeline.md) | validation, corrupt-file decisions, batches, and diagnostics |
| [Nature CNN](projects/nature-cnn.md) | reusable CNN blocks, regularization, curves, and saved artifacts |

## How this library grows

The second course in the [PyTorch for Deep Learning certificate](https://www.coursera.org/specializations/pytorch-for-deep-learning) includes model tuning, TorchVision, and transfer learning. The [reference notes](reference/index.md#course-2-what-changes-in-practice) cover the labs studied so far; a full new guide can follow when the material is complete. This library uses original explanations rather than reproducing course exercises.

For a short review outside the documentation site, open [DaZu's PyTorch guides](https://dazu.xyz/learn/pytorch/).
