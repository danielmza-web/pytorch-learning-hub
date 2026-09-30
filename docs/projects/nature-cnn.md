---
title: Nature CNN
study_context: PyTorch Fundamentals
tags:
  - project
  - cifar100
  - overfitting
last_reviewed: 2026-10-01
---

# Project: Nature CNN and overfitting

Use one image model to connect reusable blocks, learning curves, and the difference between fitting and generalizing.
{ .page-lead }

## The question

How do reusable CNN blocks and validation curves help you separate genuine learning from memorization?

## What to remember

Regularization is not one setting. It is the combination of a stable data split, training-only augmentation, model capacity, weight decay, dropout, and selecting a checkpoint by validation behavior.

## Key code

```python
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2),
        )
```

The reusable block makes each transformation explicit: learn local patterns, normalize activation statistics, add nonlinearity, then reduce spatial size. Reuse makes shape debugging and comparisons easier.

**Check yourself:** if training loss falls while validation loss rises, which change comes first? Inspect the split, class mistakes, and augmentation before making the CNN deeper. A deeper model can fit training data even more closely without improving unseen images.

## Evidence and visual

The full project downloads public CIFAR-100 through TorchVision, filters 15 nature classes, trains this CNN, and writes its own prediction grid, confusion matrix, curves, checkpoint, and JSON metrics. The architecture map explains the shape contract before a run is started.

<div class="recall-flow" role="group" aria-label="Input to output">
<div><b>RGB batch</b><code>[N,3,32,32]</code><small>keep the grid</small></div>
<div><b>Block 1</b><code>[N,32,16,16]</code><small>conv, BN, ReLU, pool</small></div>
<div><b>Block 2</b><code>[N,64,8,8]</code><small>learn higher-level features</small></div>
<div><b>Block 3</b><code>[N,128,4,4]</code><small>flatten to 2048 features</small></div>
<div><b>Classifier</b><code>[N,15]</code><small>15 nature classes</small></div>
</div>
<p class="visual-caption">Illustration: shapes and operations, not measured model performance.</p>


## Interactive check

<div class="interactive-panel curve-lab" data-curve-lab>
  <div class="interactive-heading">Illustrative train–validation gap</div>
  <label>Illustrative regularization strength <input type="range" min="0" max="100" value="35" data-regularization></label>
  <canvas width="560" height="240" data-curve-canvas aria-label="Illustrative training and validation loss curves, not measured Nature CNN data"></canvas>
  <output data-curve-output aria-live="polite"></output>
  <small>This visual explains a pattern to look for; it is not measured CIFAR-100 performance.</small>
</div>

## Run it yourself

```bash
python examples/nature_cnn.py
```

The default small run uses 120 images per selected class for four epochs and writes artifacts to `artifacts/nature-cnn/`. It uses CUDA if available; add `--device cpu` to force CPU. For a longer GPU run over the complete selected dataset:

```bash
python examples/nature_cnn.py --full --device cuda
```

Use `--smoke-test` to verify the model shape without downloading data.

## Validation and final test

The maintained script now splits the training pool with a fixed seed (`--validation-fraction 0.2`). It chooses the minimum validation-loss checkpoint, restores those weights, then evaluates the official test split at the end. The saved report records disjoint source indices and `best_epoch`; the checkpoint records normalization and class order. Early stopping would end training earlier and is a separate decision.

The default 1,800-image training pool becomes 1,440 training and 360 validation images. Validation uses deterministic preprocessing on the same training source, without random flips. The curve control above is an illustration, not a retained CIFAR-100 measurement. A full dataset run remains optional.

## Complete source

??? note "Open the maintained runnable script"
    ```python
    --8<-- "examples/nature_cnn.py"
    ```

The script imports the shared checkpoint/split helper from `examples/validation_patterns.py`; keep both files when running outside a full checkout. Train and validation share decoded image storage but use independent transform policies.

??? note "Shared deterministic split and best-checkpoint helper"
    ```python
    --8<-- "examples/validation_patterns.py"
    ```
