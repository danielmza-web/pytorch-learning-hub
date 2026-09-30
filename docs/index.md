---
title: PyTorch learning hub
tags:
  - home
last_reviewed: 2026-09-30
---

<div class="hub-hero" markdown="1">

<span class="eyebrow">A practical PyTorch reference</span>

# Understand the step. Remember the reason.

Follow a model from its first tensor to a trustworthy result. Use the short explanations when learning and return to a specific question when coding.

Start with the working path, then follow the question you need: improve training, prepare image data, reuse a model, or classify text. Each guide connects the idea, important functions and parameters, a small example, and a check for understanding.

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

Prepare realistic variation, inspect noise, and reuse a pretrained model.

[Transforms and noise](guides/vision/augmentation.md) · [Review CNNs](guides/fundamentals/vision-real-data.md)

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

<div class="topic-card" markdown="1">

**I want better or faster training**

Choose metrics, schedules and search settings; measure loading, compute and memory.

[Metrics and tuning](guides/training/training-quality.md) · [Efficient pipelines](guides/training/efficient-training.md)

</div>

<div class="topic-card" markdown="1">

**I'm working with text**

Connect token IDs, masks and embeddings to a pooled classifier or DistilBERT.

[Tokens and embeddings](guides/text/tokens-embeddings.md) · [Text classifiers](guides/text/text-classifiers.md)

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

## Continue through connected guides

| Guide | First understand | Then apply |
| --- | --- | --- |
| Fundamentals | [Tensors and training](guides/fundamentals/core-workflow.md) | [CNNs, data checks and saving](guides/fundamentals/vision-real-data.md) |
| Training | [Metrics, schedules and Optuna](guides/training/training-quality.md) | [DataLoader, Lightning, profiling and memory](guides/training/efficient-training.md) |
| Vision | [Datasets, transforms and noise](guides/vision/augmentation.md) | [Pretrained inference and transfer learning](guides/vision/pretrained-models.md) |
| Text | [Tokens, masks and embeddings](guides/text/tokens-embeddings.md) | [Pooling, imbalance and fine-tuning](guides/text/text-classifiers.md) |

Read the mechanism first; use the application page when deciding how to implement it. The sidebar follows this order. If you remember a function name instead of a topic, open the [function finder](reference/index.md#function-finder).

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
| How to simulate scattered faulty pixels | [Noise and the visual comparison](guides/vision/augmentation.md#noise-as-a-controlled-augmentation) |
| Which LR scheduler to step after validation | [Schedulers](guides/training/training-quality.md#learning-rate-schedulers) |
| Why a GPU waits for data | [DataLoader settings](guides/training/efficient-training.md#dataloader-settings) |
| How accumulation changes samples per update | [Gradient accumulation](guides/training/efficient-training.md#gradient-accumulation) |
| Why padding changes a pooled text vector | [Masked pooling](guides/text/text-classifiers.md#manual-pooling-must-ignore-padding) |
| What frozen pretrained layers still do | [Transfer strategies](guides/vision/pretrained-models.md#three-transfer-learning-strategies) |

## Learn from complete examples

The four projects are maintained programs, not isolated fragments. Each page connects one question to code, evidence, a small interactive check, and complete source. The image projects use limited data by default and choose CUDA when available; `--device cpu` forces a CPU run.

| Project | What it makes visible |
| --- | --- |
| [Nonlinear regression](projects/regression.md) | why an activation changes what a network can represent |
| [EMNIST letter classifier](projects/emnist.md) | a complete image-classification workflow and CNN shape path |
| [Robust image pipeline](projects/robust-image-pipeline.md) | validation, corrupt-file decisions, batches, and diagnostics |
| [Nature CNN](projects/nature-cnn.md) | reusable CNN blocks, regularization, curves, and saved artifacts |

## How this library grows

The topic map draws on the first two courses in the [PyTorch for Deep Learning certificate](https://www.coursera.org/professional-certificates/pytorch-for-deep-learning): foundations, model tuning, TorchVision, text models and efficient pipelines. The [study map and sources](about/index.md#study-map) explain where each group connects. Explanations and examples are original; course exercises and solutions remain outside this library.

For a short review outside the documentation site, open [DaZu's PyTorch guides](https://dazu.xyz/learn/pytorch/).
