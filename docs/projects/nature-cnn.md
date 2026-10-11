---
title: P4 · Nature CNN
study_context: PyTorch Fundamentals
tags:
  - project
  - cifar100
  - overfitting
project_id: P4
last_reviewed: 2026-10-10
---

# P4 · Nature CNN {#project-nature-cnn-and-overfitting}

Use one image model to connect reusable blocks, learning curves, and the difference between fitting and generalizing.
{ .page-lead }

<div class="learning-context" data-project-id="P4" markdown="1">

**P4 · Nature CNN**
{ .learning-location }

**You will practice:** Connect reusable CNN blocks with training/validation curves and checkpoint selection.

**Read first:** [1.2 · Images and CNNs](../guides/fundamentals/vision-real-data.md)

**Data and setup:** Public CIFAR-100 filtered to 15 nature classes; full runs download missing data. The shape smoke check is offline.

</div>

## The question

How do reusable CNN blocks and validation curves help you separate genuine learning from memorization?

## What to remember

Regularization is not one setting. It is the combination of a stable data split, training-only augmentation, model capacity, weight decay, dropout, and selecting a checkpoint by validation behavior.

CNN means convolutional neural network; BN means batch normalization and ReLU is the rectified linear unit activation. The logits `[N,15]` have one score per selected nature class. **Common trap:** the three curve diagrams illustrate patterns; they do not run CIFAR-100 training or prove a better hyperparameter.

## Key code

**Code key:** the short snippets are excerpts from the maintained script, not standalone programs. The complete source below supplies imports, classes, data and helper functions; run that script from the repository root.

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


## Visual check {#interactive-check}

<!-- learning-overview:curves -->
<section class="learning-overview" data-overview="curves" aria-label="Compare the train–validation relationship"><p><strong>Compare the train–validation relationship</strong></p><div class="overview-grid"><figure><figcaption>Underfitting</figcaption><span class="plot-label">Loss</span><div class="overview-plot"><div class="plot-ticks" aria-hidden="true"><span>1.6</span><span>0.8</span><span>0</span></div><svg class="overview-chart" viewBox="0 0 280 150" role="img" aria-label="Underfitting: both losses remain high. Same loss axis: 0 to 1.6; illustrative epochs 0 to 29."><path d="M8 10V140H272" fill="none" stroke="currentColor"/><path d="M8,18.45 L17.1034,26.3013 L26.2069,33.141 L35.3103,39.0994 L44.4138,44.2902 L53.5172,48.8122 L62.6207,52.7515 L71.7241,56.1833 L80.8276,59.173 L89.931,61.7775 L99.0345,64.0464 L108.1379,66.0229 L117.2414,67.7448 L126.3448,69.2449 L135.4483,70.5517 L144.5517,71.6901 L153.6552,72.6818 L162.7586,73.5458 L171.8621,74.2985 L180.9655,74.9541 L190.069,75.5253 L199.1724,76.0229 L208.2759,76.4564 L217.3793,76.8341 L226.4828,77.1631 L235.5862,77.4497 L244.6897,77.6993 L253.7931,77.9168 L262.8966,78.1063 L272,78.2714" fill="none" stroke="#007995" stroke-width="3"/><path d="M8,15.2 L17.1034,21.6718 L26.2069,27.4277 L35.3103,32.5468 L44.4138,37.0995 L53.5172,41.1486 L62.6207,44.7498 L71.7241,47.9525 L80.8276,50.8009 L89.931,53.3343 L99.0345,55.5873 L108.1379,57.5911 L117.2414,59.3732 L126.3448,60.9582 L135.4483,62.3678 L144.5517,63.6215 L153.6552,64.7365 L162.7586,65.7281 L171.8621,66.61 L180.9655,67.3944 L190.069,68.092 L199.1724,68.7124 L208.2759,69.2642 L217.3793,69.7549 L226.4828,70.1913 L235.5862,70.5795 L244.6897,70.9247 L253.7931,71.2317 L262.8966,71.5048 L272,71.7477" fill="none" stroke="#bf5a0c" stroke-width="3" stroke-dasharray="6 3"/></svg></div><div class="plot-epochs"><span>0</span><span>Illustrative epoch</span><span>29</span></div><p>Underfitting: both losses remain high.</p><p>Final train 0.7597; validation 0.84.</p></figure><figure><figcaption>Balanced illustration</figcaption><span class="plot-label">Loss</span><div class="overview-plot"><div class="plot-ticks" aria-hidden="true"><span>1.6</span><span>0.8</span><span>0</span></div><svg class="overview-chart" viewBox="0 0 280 150" role="img" aria-label="Balanced illustration: both losses fall, with a modest gap. Same loss axis: 0 to 1.6; illustrative epochs 0 to 29."><path d="M8 10V140H272" fill="none" stroke="currentColor"/><path d="M8,72.5625 L17.1034,80.4138 L26.2069,87.2535 L35.3103,93.2119 L44.4138,98.4027 L53.5172,102.9247 L62.6207,106.864 L71.7241,110.2958 L80.8276,113.2855 L89.931,115.89 L99.0345,118.1589 L108.1379,120.1354 L117.2414,121.8573 L126.3448,123.3574 L135.4483,124.6642 L144.5517,125.8026 L153.6552,126.7943 L162.7586,127.6583 L171.8621,128.411 L180.9655,129.0666 L190.069,129.6378 L199.1724,130.1354 L208.2759,130.5689 L217.3793,130.9466 L226.4828,131.2756 L235.5862,131.5622 L244.6897,131.8118 L253.7931,132.0293 L262.8966,132.2188 L272,132.3839" fill="none" stroke="#007995" stroke-width="3"/><path d="M8,69.3125 L17.1034,75.7843 L26.2069,81.5402 L35.3103,86.6593 L44.4138,91.212 L53.5172,95.2611 L62.6207,98.8623 L71.7241,102.065 L80.8276,104.9134 L89.931,107.4468 L99.0345,109.6998 L108.1379,111.7036 L117.2414,113.4857 L126.3448,115.0707 L135.4483,116.4803 L144.5517,117.734 L153.6552,118.849 L162.7586,119.8406 L171.8621,120.7225 L180.9655,121.5069 L190.069,122.2045 L199.1724,122.8249 L208.2759,123.3767 L217.3793,123.8674 L226.4828,124.3038 L235.5862,124.692 L244.6897,125.0372 L253.7931,125.3442 L262.8966,125.6173 L272,125.8602" fill="none" stroke="#bf5a0c" stroke-width="3" stroke-dasharray="6 3"/></svg></div><div class="plot-epochs"><span>0</span><span>Illustrative epoch</span><span>29</span></div><p>Balanced illustration: both losses fall, with a modest gap.</p><p>Final train 0.0937; validation 0.174.</p></figure><figure><figcaption>Overfitting</figcaption><span class="plot-label">Loss</span><div class="overview-plot"><div class="plot-ticks" aria-hidden="true"><span>1.6</span><span>0.8</span><span>0</span></div><svg class="overview-chart" viewBox="0 0 280 150" role="img" aria-label="Overfitting: training falls while validation turns upward. Same loss axis: 0 to 1.6; illustrative epochs 0 to 29."><path d="M8 10V140H272" fill="none" stroke="currentColor"/><path d="M8,72.5625 L17.1034,80.4138 L26.2069,87.2535 L35.3103,93.2119 L44.4138,98.4027 L53.5172,102.9247 L62.6207,106.864 L71.7241,110.2958 L80.8276,113.2855 L89.931,115.89 L99.0345,118.1589 L108.1379,120.1354 L117.2414,121.8573 L126.3448,123.3574 L135.4483,124.6642 L144.5517,125.8026 L153.6552,126.7943 L162.7586,127.6583 L171.8621,128.411 L180.9655,129.0666 L190.069,129.6378 L199.1724,130.1354 L208.2759,130.5689 L217.3793,130.9466 L226.4828,131.2756 L235.5862,131.5622 L244.6897,131.8118 L253.7931,132.0293 L262.8966,132.2188 L272,132.3839" fill="none" stroke="#007995" stroke-width="3"/><path d="M8,69.3125 L17.1034,75.7843 L26.2069,81.5402 L35.3103,86.6593 L44.4138,91.212 L53.5172,95.2611 L62.6207,98.8623 L71.7241,102.065 L80.8276,104.9134 L89.931,107.4468 L99.0345,109.6998 L108.1379,111.7036 L117.2414,113.4703 L126.3448,114.8813 L135.4483,115.9238 L144.5517,116.6172 L153.6552,116.9786 L162.7586,117.0234 L171.8621,116.7653 L180.9655,116.2165 L190.069,115.3876 L199.1724,114.2883 L208.2759,112.9272 L217.3793,111.3118 L226.4828,109.4489 L235.5862,107.3445 L244.6897,105.0039 L253.7931,102.4319 L262.8966,99.6328 L272,96.6102" fill="none" stroke="#bf5a0c" stroke-width="3" stroke-dasharray="6 3"/></svg></div><div class="plot-epochs"><span>0</span><span>Illustrative epoch</span><span>29</span></div><p>Overfitting: training falls while validation turns upward.</p><p>Final train 0.0937; validation 0.534.</p></figure></div><p><span class="curve-key train-key">Solid teal: training</span> · <span class="curve-key validation-key">Dashed orange: validation</span>. Read each pair on the same axes; these are three situations, not a continuous regularization law.</p><p><small>Synthetic curves, not measured Nature CNN or other model results.</small></p></section>
<!-- /learning-overview -->

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

**Inspect after running:** Inspect the validation-selected checkpoint and final test report in artifacts/nature-cnn/; the three curve diagrams are illustrations.

**Connect the result:** [1.2 · Images and CNNs](../guides/fundamentals/vision-real-data.md). [Choose another project](index.md).
