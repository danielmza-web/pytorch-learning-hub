---
title: Complete PyTorch projects
tags:
  - projects
last_reviewed: 2026-09-27
---

# Complete PyTorch projects

Choose the behavior you want to understand. Each project connects one idea to a maintained script, visible output, and the guide section that explains it.
{ .page-lead }

## Choose a project

<div class="topic-grid" markdown="1">

<div class="topic-card" markdown="1">

**01 · Nonlinear regression**

Why can a network fit a curve that a linear model misses? Compare a measured prediction chart and validation MSE.

[Open project](regression.md) · [Review activations](../guides/fundamentals/core-workflow.md#models-activations-and-logits)

</div>

<div class="topic-card" markdown="1">

**02 · EMNIST letters**

Why preserve an image's spatial structure? Trace the CNN, then inspect predictions, curves, and a confusion matrix.

[Open project](emnist.md) · [Review CNN shapes](../guides/fundamentals/vision-real-data.md#cnn-architecture-and-shapes)

</div>

<div class="topic-card" markdown="1">

**03 · Robust image pipeline**

How should bad files be handled before training? Follow a validated manifest through batches and diagnostics.

[Open project](robust-image-pipeline.md) · [Review reliable data](../guides/fundamentals/vision-real-data.md#reliable-image-data)

</div>

<div class="topic-card" markdown="1">

**04 · Nature CNN**

How can validation reveal overfitting? Compare reusable blocks, regularization, and learning curves.

[Open project](nature-cnn.md) · [Review generalization](../guides/fundamentals/vision-real-data.md#generalization-and-regularization)

</div>

</div>

## What each project includes

Every project starts with a question and a short explanation, then shows the key code, evidence or a clearly labelled illustration, an interactive check, run instructions, and the maintained complete source.

## Small by default, longer when requested

The default commands use fixed seeds and intentionally limited data or epochs. Regression runs on CPU. The three image projects use `--device auto`: they select CUDA when it is available and otherwise run on CPU. Add `--device cpu` to compare on CPU deliberately. The optional `--full --device cuda` mode increases the run budget without changing the conceptual workflow.

!!! note "Results policy"
    Only output produced by retained, reproducible runs is reported as measurement. Conceptual diagrams and adjustable curves are clearly labelled illustrative. Smoke tests verify code paths; they are not model-quality benchmarks.
