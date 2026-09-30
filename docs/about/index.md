---
title: About and sources
tags:
  - about
  - sources
last_reviewed: 2026-09-30
---

# About and sources

Daniel Zurita's public PyTorch learning reference connects explanations, visual checks, and runnable projects. It is an independent companion for understanding concepts, not an official course site.
{ .page-lead }

## Design principles

- Explain the mechanism before adding abstractions.
- Keep shapes and data flow visible.
- Prefer fewer substantial guides over repeated small pages.
- Keep default project runs small enough for an ordinary CPU computer.
- Separate smoke tests, illustrative diagrams, and measured model results.
- Explain failures and limitations, not only successful outputs.
- Publish a future guide only when it contains useful material.
- Store no accounts, analytics, progress, or personal learning history.

## How the library grows

Each PyTorch guide uses two substantial pages: mechanisms and applied decisions. Four guides connect Fundamentals, Training, Vision and Text. The same concepts recur deliberately: shape contracts, split quality, optimizer ownership, evaluation and resource measurements. The function finder provides a direct route back to code.

The compact guide index and short reviews live at [dazu.xyz/learn/pytorch/](https://dazu.xyz/learn/pytorch/). This detailed library remains separate so it can grow without adding a build process to DaZu.

## Primary technical sources

- [PyTorch documentation](https://docs.pytorch.org/docs/stable/)
- [TorchVision documentation](https://docs.pytorch.org/vision/stable/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index)
- [Optuna](https://optuna.readthedocs.io/en/stable/)
- [Lightning](https://lightning.ai/docs/pytorch/stable/)
- [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/)
- [Material for MkDocs documentation](https://squidfunk.github.io/mkdocs-material/)
- [EMNIST Letters dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EMNIST.html)
- [CIFAR-10 dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
- [CIFAR-100 dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html)

## Study context and content boundary

The topic sequence is informed by DeepLearning.AI's [PyTorch for Deep Learning certificate](https://www.coursera.org/professional-certificates/pytorch-for-deep-learning), especially its first two courses. The 2026-09-30 content review considered the local presentations, lab exports, notebooks and assessment topic outlines. The guides synthesize mechanisms with original examples; they do not redistribute those source files.

This site does not reproduce Coursera assessments, protected notebooks, quizzes, or graded solutions. Explanations, diagrams, examples, and project structures are original. Course names belong to their respective owners and are used descriptively.

## Study map

Use the course/module names only as an orientation to topics. The public library is arranged by practical questions rather than exercise order.

| Study context | Concepts carried into this library | Start here |
| --- | --- | --- |
| Fundamentals · module 1 | regression, features, activations, tensor creation/manipulation/math | [Core workflow](../guides/fundamentals/core-workflow.md) |
| Fundamentals · module 2 | batching, classification, losses, gradients, optimizers, devices | [Training loop](../guides/fundamentals/core-workflow.md#the-complete-training-loop) |
| Fundamentals · module 3 | lazy access, labels, resizing, normalization, split transforms, corrupt files | [Reliable image data](../guides/fundamentals/vision-real-data.md#reliable-image-data) |
| Fundamentals · module 4 | CNNs, pooling, modularity, inspection, dropout, BatchNorm, overfitting | [Vision fundamentals](../guides/fundamentals/vision-real-data.md) |
| Techniques and Ecosystem Tools · module 1 | metrics, hyperparameters, schedules, flexible architectures, Optuna, quality/cost | [Metrics and tuning](../guides/training/training-quality.md) |
| Techniques and Ecosystem Tools · module 2 | TorchVision datasets/utilities, transforms, impulse noise, task outputs, transfer | [Transforms and noise](../guides/vision/augmentation.md) |
| Techniques and Ecosystem Tools · module 3 | tokenization, representations, GloVe/context, pooled classifiers, imbalance, DistilBERT | [Tokens and embeddings](../guides/text/tokens-embeddings.md) |
| Techniques and Ecosystem Tools · module 4 | loader settings, Lightning, profiling, AMP, accumulation, callbacks | [Efficient pipelines](../guides/training/efficient-training.md) |

Detection and segmentation are covered as pretrained inference and output interpretation. Broader deployment and advanced-architecture topics need their own source review before adding a guide. Extra comparisons such as Gaussian versus impulse noise are labelled as explanatory extensions.

## Visuals and measurements

- Interactive diagrams are implemented locally with HTML, CSS, and JavaScript.
- Conceptual charts are explicitly labelled illustrative.
- Model-quality metrics are published only when generated by retained, reproducible experiments.
- Complete project code is included from the maintained scripts rather than copied by hand.
- The DaZu mark is used with the owner's permission.
