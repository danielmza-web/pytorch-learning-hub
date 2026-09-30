---
title: Start here
tags:
  - home
last_reviewed: 2026-09-30
---

# PyTorch Learning Hub

A library of explanations, visual checks and runnable examples for learning PyTorch or remembering how to use it. It covers the foundations from Course 1 and the training, vision and text workflows from Course 2.
{ .page-lead }

## What you will find here

**Guides:** understand a concept and connect its steps. Four topics, two pages each, with shapes, functions, parameters and common mistakes.

[**Projects**](projects/index.md): run the code and inspect the result. Seven original scripts with important excerpts, run commands, output explanations and complete source.

[**Quick reference**](reference/index.md): look up a function or error. Use the function finder, training cheatsheet, shape reminders and troubleshooting.

## Choose a guide

Each topic has a first page for the main ideas and a second page for applying them. The sidebar keeps this order throughout the library.

<div class="topic-grid" markdown="1">

<div class="topic-card" markdown="1">

**1 · Fundamentals**

Tensors, models, the learning loop and image classification.

[Build and train](guides/fundamentals/core-workflow.md) · [Images and CNNs](guides/fundamentals/vision-real-data.md)

</div>

<div class="topic-card" markdown="1">

**2 · Training**

Validation metrics, tuning, data loading and memory use.

[Metrics and tuning](guides/training/training-quality.md) · [Efficient pipelines](guides/training/efficient-training.md)

</div>

<div class="topic-card" markdown="1">

**3 · Vision**

Image variation, noise, pretrained outputs and adapting models.

[Transforms and noise](guides/vision/augmentation.md) · [Pretrained models](guides/vision/pretrained-models.md)

</div>

<div class="topic-card" markdown="1">

**4 · Text**

Tokenization, embeddings, variable lengths and text classification.

[Tokens and embeddings](guides/text/tokens-embeddings.md) · [Classifiers and fine-tuning](guides/text/text-classifiers.md)

</div>

</div>

**If you are starting again:** read Fundamentals first. Training explains how to compare and improve the same loop; Vision and Text show how different inputs and pretrained models fit into it. You can jump directly to a topic when reviewing.

## Selected Course 2 examples

These three projects combine the most useful mechanisms across several labs. They are small original examples, with offline defaults, so you can inspect the workflow without downloading a dataset first.

- [**Controlled training comparison**](projects/training-comparison.md): shared starting weights and split, macro F1, learning-rate scheduling and gradient accumulation.
- [**Image augmentation and head training**](projects/vision-head.md): noise before normalization, a replacement ResNet head, frozen features and saved class metadata.
- [**Variable-length text classifier**](projects/text-bags.md): a training-only vocabulary, token offsets, pooled embeddings and class weights.

The project gallery also contains the four foundation examples: regression, EMNIST letters, robust image loading and a regularized CNN. Choose a project after reading its short “What to remember” section; the full source is expandable.

## How to use a page

Read the explanation, follow the important code, then use its visual check or run command. Return through the sidebar for another topic, or search for an exact name such as `EmbeddingBag`, `ReduceLROnPlateau` or `pin_memory`.

Illustrations are labelled; toy runs demonstrate mechanisms, and retained measurements state their source. This is an independent study library with original content. [About and sources](about/index.md) explains its scope and evidence policy.
