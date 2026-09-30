---
title: Complete PyTorch projects
tags:
  - projects
last_reviewed: 2026-10-01
---

# Complete PyTorch projects

Choose the workflow you want to run. Each page explains the important steps before showing the complete maintained script.
{ .page-lead }

## Selected Course 2 workflows

Three examples combine useful ideas across the course rather than reproduce each lab.

<div class="project-grid" markdown="1">

<div class="project-choice" markdown="1">

[Controlled training comparison](training-comparison.md)

Metrics, learning-rate choice, plateau scheduling, gradient accumulation

**Run / output:** Small synthetic classification problem on CPU; saves a comparison report.

</div>

<div class="project-choice" markdown="1">

[Image augmentation and head training](vision-head.md)

Training transforms, impulse noise, frozen ResNet backbone, replacement head

**Run / output:** Synthetic images with random weights; no downloads. Optional own images + pretrained weights.

</div>

<div class="project-choice" markdown="1">

[Variable-length text classifier](text-bags.md)

Training-only vocabulary, offsets, mean pooling, class weights

**Run / output:** Tiny original phrases on CPU; saves predictions and the model contract.

</div>

</div>

## Foundation projects

<div class="project-grid" markdown="1">

<div class="project-choice" markdown="1">

[Nonlinear regression](regression.md)

Linear versus nonlinear representation, MSE, autograd

**Run / output:** Retained prediction chart and validation MSE.

</div>

<div class="project-choice" markdown="1">

[EMNIST letters](emnist.md)

CNN shapes, classification, evaluation

**Run / output:** Predictions, learning curves and a confusion matrix.

</div>

<div class="project-choice" markdown="1">

[Robust image pipeline](robust-image-pipeline.md)

Stable class IDs, readable files, rejected samples

**Run / output:** Validated manifest and diagnostics.

</div>

<div class="project-choice" markdown="1">

[Nature CNN](nature-cnn.md)

Reusable blocks, regularization, overfitting

**Run / output:** Training versus validation curves.

</div>

</div>

## Running and interpreting the examples

Commands run from the repository root after preparing the documented project dependencies. On Windows, use `py -3.14` in place of `python` if that is your installed launcher.

The three new workflows use CPU and small offline defaults. The original image projects use `--device auto`, selecting CUDA when available and otherwise CPU; they may download public TorchVision data. Their optional `--full --device cuda` mode increases the run budget.

!!! note "What a small run proves"
    Synthetic images, toy phrases and smoke tests verify the code path and make the mechanics visible. They do not establish real dataset quality, pretrained transfer performance or an efficiency benchmark. A measured claim needs retained evidence from the corresponding run.
