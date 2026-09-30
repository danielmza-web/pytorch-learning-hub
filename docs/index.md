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

| Part of the library | Use it when… | What it contains |
| --- | --- | --- |
| **Guides** | You need to understand a concept or connect several steps. | Four topics, two connected pages each: explanations, shapes, functions, parameters and common mistakes. |
| [**Projects**](projects/index.md) | You want to run the code and inspect what happens. | Seven original scripts with important excerpts, run commands, output explanations and complete source. |
| [**Quick reference**](reference/index.md) | You remember a name or error and need a direct answer. | A function finder, training cheatsheet, shape reminders and troubleshooting. |

## Choose a guide

Each topic has a first page for the main ideas and a second page for applying them. The sidebar keeps this order throughout the library.

| Topic | First page | Continue with |
| --- | --- | --- |
| **1 · Fundamentals** | [Tensors, models and the training loop](guides/fundamentals/core-workflow.md) | [Image data, CNNs and validation](guides/fundamentals/vision-real-data.md) |
| **2 · Training** | [Metrics, schedules and tuning](guides/training/training-quality.md) | [Data loading, profiling and memory](guides/training/efficient-training.md) |
| **3 · Vision** | [Transforms, augmentation and noise](guides/vision/augmentation.md) | [Pretrained models and transfer learning](guides/vision/pretrained-models.md) |
| **4 · Text** | [Tokens, padding and embeddings](guides/text/tokens-embeddings.md) | [Pooled classifiers and fine-tuning](guides/text/text-classifiers.md) |

**If you are starting again:** read Fundamentals first. Training explains how to compare and improve the same loop; Vision and Text show how different inputs and pretrained models fit into it. You can jump directly to a topic when reviewing.

## Selected Course 2 examples

These three projects combine the most useful mechanisms across several labs. They are small original examples, with offline defaults, so you can inspect the workflow without downloading a dataset first.

| Project | What you can run and remember |
| --- | --- |
| [Controlled training comparison](projects/training-comparison.md) | Compare learning rates using the same starting weights and split; read macro F1; schedule LR after validation; accumulate gradients. |
| [Image augmentation and head training](projects/vision-head.md) | Apply noise before normalization, replace a ResNet head, keep the backbone frozen, and save class/preprocessing metadata. |
| [Variable-length text classifier](projects/text-bags.md) | Build a vocabulary from training text, collate token IDs and offsets, pool embeddings, and use class weights. |

The project gallery also contains the four foundation examples: regression, EMNIST letters, robust image loading and a regularized CNN. Choose a project after reading its short “What to remember” section; the full source is expandable.

## How to use a page

Read the explanation, follow the important code, then use its visual check or run command. Return through the sidebar for another topic, or search for an exact name such as `EmbeddingBag`, `ReduceLROnPlateau` or `pin_memory`.

Illustrations are labelled; toy runs demonstrate mechanisms, and retained measurements state their source. This is an independent study library with original content. [About and sources](about/index.md) explains its scope and evidence policy.
