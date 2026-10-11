---
title: P3 · Robust image pipeline
study_context: PyTorch Fundamentals
tags:
  - project
  - data-pipeline
project_id: P3
last_reviewed: 2026-10-10
---

# P3 · Robust image pipeline {#project-robust-image-pipeline}

Make the input contract visible before training: class order, readable files, rejected samples, and stable validation data.
{ .page-lead }

<div class="learning-context" data-project-id="P3" markdown="1">

**P3 · Robust image pipeline**
{ .learning-location }

**You will practice:** Make readable files, rejected samples and stable class IDs explicit before training.

**Read first:** [1.1 · Build and train](../guides/fundamentals/core-workflow.md) · [1.2 · Images and CNNs](../guides/fundamentals/vision-real-data.md)

**Data and setup:** Public CIFAR-10, materialized into five class folders with a deliberate corrupt file. The smoke check uses local temporary files.

</div>

## The question

How can a dataset tell you about bad input before a training run turns it into a vague error or misleading metric?

## What to remember

Data quality is part of model quality. Define class ids deterministically, inspect files when indexing, record failures, and keep random augmentation outside validation.

Here `[N,3,32,32]` means N RGB images, three channels and a 32-pixel square; RGB means red/green/blue. File validation detects damaged input, not incorrect labels. **Common trap:** a folder scan can be valid while a label is wrong, or a file changes after indexing; inspect rejections and verify the sample again when loading fails.

## Key code

**Code key:** the short snippets are excerpts from the maintained script, not standalone programs. The complete source below supplies imports, classes, data and helper functions; run that script from the repository root.

```python
self.class_names = sorted(path.name for path in self.root.iterdir() if path.is_dir())
self.class_to_index = {name: index for index, name in enumerate(self.class_names)}

try:
    with Image.open(path) as image:
        image.verify()
except Exception as error:
    self.invalid.append(InvalidImage(path, type(error).__name__))
    continue
```

Sorting protects the class-to-index contract. `verify()` checks readability during indexing, so a corrupt file is reported before a batch reaches the model.

**Check yourself:** if class ids come from filesystem iteration order, two runs can disagree about what output index `0` means. Sorting once and saving that mapping makes predictions interpretable after training.

## Evidence and visual

The full project downloads public CIFAR-10 images through TorchVision, writes a deterministic five-class folder dataset, deliberately includes one corrupt JPEG, and then trains only on the validated samples. It saves a validation report, prediction grid, confusion matrix, curves, checkpoint, and JSON metrics.

<div class="recall-flow" role="group" aria-label="Input to output">
<div><b>Scan folders</b><code>class → stable ID</code><small>sort and retain labels</small></div>
<div><b>Validate</b><code>readable image?</code><small>accepted samples or rejected-file report</small></div>
<div><b>Prepare</b><code>[3,32,32]</code><small>decode RGB and transform</small></div>
<div><b>Batch</b><code>[N,3,32,32]</code><small>DataLoader groups accepted samples</small></div>
</div>
<p class="visual-caption">Illustration: shapes and operations, not measured model performance.</p>


## Visual check {#interactive-check}

<!-- learning-overview:pipeline -->
<section class="learning-overview" data-overview="pipeline" aria-label="See all three manifest decisions"><p><strong>See all three manifest decisions</strong></p><div class="overview-grid"><div class="overview-card"><b>rose/valid.png</b><p><strong>Accept → manifest → batch</strong></p><p>Readable image, declared extension and known class. Retain stable path and class ID.</p></div><div class="overview-card"><b>bee/broken.jpg</b><p><strong>Reject → diagnostics</strong></p><p>Decoding fails. Keep the path and reason in the invalid-file report; do not batch it.</p></div><div class="overview-card"><b>notes/readme.txt</b><p><strong>Skip → extension policy</strong></p><p>Not a declared image type. Do not interpret it as training data.</p></div></div><p><small>Original illustration; not measured model performance.</small></p></section>
<!-- /learning-overview -->

## Run it yourself

```bash
python examples/robust_dataset.py
```

The default small run builds 400 real image files and trains for five epochs. It uses CUDA if available; add `--device cpu` to force CPU. For a longer GPU run with a larger materialized dataset:

```bash
python examples/robust_dataset.py --full --device cuda
```

Use `--smoke-test` to verify corrupt-file handling without downloading data.

## Complete source

??? note "Open the maintained runnable script"
    ```python
    --8<-- "examples/robust_dataset.py"
    ```

**Inspect after running:** Inspect the rejected-file report, saved class mapping and predictions in artifacts/robust-pipeline/.

**Connect the result:** [1.1 · Build and train](../guides/fundamentals/core-workflow.md) · [1.2 · Images and CNNs](../guides/fundamentals/vision-real-data.md). [Choose another project](index.md).
