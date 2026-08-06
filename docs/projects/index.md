---
title: Complete PyTorch projects
tags:
  - projects
last_reviewed: 2026-08-06
---

# Complete PyTorch projects

These projects connect an important PyTorch idea to a program you can run. Start with the question you need to clarify; each project points back to the guide sections that explain its parts.

## How every project page works

1. **The question** identifies the transferable idea.
2. **What to remember** explains the mental model.
3. **Key code** annotates the important implementation decisions.
4. **Evidence and visual** separates reproduced output from illustrative diagrams.
5. **Interactive check** lets you change one relevant input.
6. **Run it yourself** gives a small CPU command and an optional longer GPU mode.
7. **Complete source** includes the maintained script directly from `examples/`.

## Choose a project

| Project | Question it answers | Guide connection | Output |
| --- | --- | --- | --- |
| [Nonlinear regression](regression.md) | Why can a nonlinear network fit a curve that a linear model misses? | [Models and activations](../guides/fundamentals/core-workflow.md#models-activations-and-logits) | reproduced prediction chart and validation MSE |
| [EMNIST letters](emnist.md) | Why should an image classifier preserve spatial structure? | [CNN architecture and shapes](../guides/fundamentals/vision-real-data.md#cnn-architecture-and-shapes) | predictions, curves, confusion matrix, metrics, checkpoint |
| [Robust image pipeline](robust-image-pipeline.md) | How should bad files be handled before training? | [Reliable image data](../guides/fundamentals/vision-real-data.md#reliable-image-data) | manifest, validation report, predictions, metrics, checkpoint |
| [Nature CNN](nature-cnn.md) | How do architecture and regularization affect generalization? | [Generalization and regularization](../guides/fundamentals/vision-real-data.md#generalization-and-regularization) | predictions, curves, confusion matrix, metrics, checkpoint |

## Small by default, longer when requested

The default commands use fixed seeds and intentionally limited data or epochs so the projects can run on an ordinary CPU computer. They still download real public datasets, create models, train, evaluate, and save artifacts. The optional `--full --device cuda` mode increases the run budget without changing the conceptual workflow.

!!! note "Results policy"
    Only output produced by retained, reproducible runs is reported as measurement. Conceptual diagrams and adjustable curves are clearly labelled illustrative. Smoke tests verify code paths; they are not model-quality benchmarks.
