---
title: Complete PyTorch projects
tags:
  - projects
last_reviewed: 2026-09-30
---

# Complete PyTorch projects

Choose the workflow you want to run. Each page explains the important steps before showing the complete maintained script.
{ .page-lead }

## Selected Course 2 workflows

Three examples combine useful ideas across the course rather than reproduce each lab.

| Project | Main ideas | Default run |
| --- | --- | --- |
| [Controlled training comparison](training-comparison.md) | Metrics, learning-rate choice, plateau scheduling, gradient accumulation | Small synthetic classification problem on CPU; saves a comparison report. |
| [Image augmentation and head training](vision-head.md) | Training transforms, impulse noise, frozen ResNet backbone, replacement head | Synthetic images with random weights; no downloads. Optional own images + pretrained weights. |
| [Variable-length text classifier](text-bags.md) | Training-only vocabulary, offsets, mean pooling, class weights | Tiny original phrases on CPU; saves predictions and the model contract. |

## Foundation projects

| Project | Main ideas | Output to inspect |
| --- | --- | --- |
| [Nonlinear regression](regression.md) | Linear versus nonlinear representation, MSE, autograd | Retained prediction chart and validation MSE. |
| [EMNIST letters](emnist.md) | CNN shapes, classification, evaluation | Predictions, learning curves and a confusion matrix. |
| [Robust image pipeline](robust-image-pipeline.md) | Stable class IDs, readable files, rejected samples | Validated manifest and diagnostics. |
| [Nature CNN](nature-cnn.md) | Reusable blocks, regularization, overfitting | Training versus validation curves. |

## Running and interpreting the examples

Commands run from the repository root after preparing the documented project dependencies. On Windows, use `py -3.14` in place of `python` if that is your installed launcher.

The three new workflows use CPU and small offline defaults. The original image projects use `--device auto`, selecting CUDA when available and otherwise CPU; they may download public TorchVision data. Their optional `--full --device cuda` mode increases the run budget.

!!! note "What a small run proves"
    Synthetic images, toy phrases and smoke tests verify the code path and make the mechanics visible. They do not establish real dataset quality, pretrained transfer performance or an efficiency benchmark. A measured claim needs retained evidence from the corresponding run.
