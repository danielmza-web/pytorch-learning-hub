---
title: PyTorch: start, build, inspect
tags:
  - home
last_reviewed: 2026-08-06
---

# PyTorch: start, build, inspect

This is a visual reference for the things you need to recognise in a PyTorch project. Use it to learn the fundamentals for the first time, review a concept before writing code, or trace a problem in a model you already have.

[Start with the three-part path](start/index.md){ .md-button .md-button--primary }
[Open a short reminder](courses/fundamentals/quick-review.md){ .md-button }

## Begin with a concrete question

| If you want to… | Start here | What you will see |
| --- | --- | --- |
| Understand a tensor and a simple neural network | [Tensors and a first model](start/first-model.md) | shapes, `nn.Linear`, activations, and a measured linear-vs-nonlinear example |
| Make an image classifier actually learn | [Build a complete classifier](start/first-classifier.md) | `Dataset`, `DataLoader`, logits, loss, gradients, and evaluation |
| Understand why CNNs work for images | [CNNs for images](start/cnn.md) | filters, feature maps, pooling, and changing tensor shapes |
| Find why a run is not behaving | [Common errors](reference/common-errors.md) | shapes, labels, devices, modes, loss, and memory checks |

## The classification loop in one picture

![Conceptual diagram showing a batch moving through model, logits, loss, gradients, and an optimizer update](assets/images/training-cycle-visual.png)

The loop is always the same: **batch → model → logits → loss → gradients → update**. Evaluation uses the same model but does not update its weights.

## What each part is responsible for

<div class="topic-grid" markdown>

<div class="topic-card">
<strong>Data and shapes</strong>

An image becomes a tensor such as `[batch, channels, height, width]`. Read the shape before changing layers.

[Tensors and shapes →](concepts/tensor-shapes.md)
</div>

<div class="topic-card">
<strong>Models and logits</strong>

`nn.Module` transforms a batch into raw class scores. Activations give a model nonlinear capacity.

[Models and training →](courses/fundamentals/models-training.md)
</div>

<div class="topic-card">
<strong>Learning and evaluation</strong>

Loss measures error; gradients tell the optimizer how to change weights. Validation checks whether that change generalizes.

[Evaluation and metrics →](courses/fundamentals/evaluation.md)
</div>

<div class="topic-card">
<strong>Image models</strong>

Convolution learns local patterns. Pooling trades some spatial detail for smaller, richer feature maps.

[Convolution explorer →](concepts/convolution.md)
</div>

</div>

## Learn from complete, runnable examples

Every project page shows the question it answers, selected code, visuals, a small interactive check, run instructions, and the maintained full script. The default configuration is deliberately small and reproducible on CPU; use `--full --device cuda` for a longer GPU run.

| Project | Start it when you need to see… |
| --- | --- |
| [Linear vs nonlinear regression](projects/regression.md) | why an activation changes what a model can represent |
| [EMNIST letter classifier](projects/emnist.md) | the full data-to-evaluation classification workflow |
| [Robust image pipeline](projects/robust-image-pipeline.md) | transforms, files, corrupt-image handling, and a validation manifest |
| [Nature CNN](projects/nature-cnn.md) | a CNN, regularization, learning curves, and overfitting decisions |

## Keep these nearby

- [Short PyTorch reminder](courses/fundamentals/quick-review.md)
- [Cheatsheet](reference/cheatsheet.md)
- [Common errors](reference/common-errors.md)
- [Fundamentals reference map](courses/fundamentals/index.md)
