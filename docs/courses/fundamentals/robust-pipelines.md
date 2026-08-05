---
title: Robust pipelines
study_context: PyTorch Fundamentals
topic_order: 5
tags:
  - data-quality
  - robustness
last_reviewed: 2026-08-05
---

# Robust real-world pipelines

A model cannot compensate for inconsistent labels, leakage, corrupt files, or transforms applied to the wrong split.

## Validate the index first

Before training, verify:

- Every path exists.
- Every label is inside the expected range.
- Class names map to stable integer ids.
- Train, validation, and test samples do not overlap.
- Image modes are converted consistently.
- Corrupt samples are logged rather than silently hidden.

```python
def open_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")
```

## Handle failures deliberately

Do not recursively skip to a new sample inside `__getitem__` without a limit; a cluster of corrupt files can create infinite recursion. Prefer validating the dataset index before training or returning a controlled sentinel with a custom `collate_fn`.

```python
def collate_valid(batch):
    valid = [sample for sample in batch if sample is not None]
    if not valid:
        raise RuntimeError("Batch contains no valid samples")
    return torch.utils.data.default_collate(valid)
```

## Reproducible splits

```python
generator = torch.Generator().manual_seed(42)
train_set, validation_set = random_split(
    dataset,
    [train_size, validation_size],
    generator=generator,
)
```

Record the split seed and class distribution. For entity-related data—multiple images of one object or person—split by entity, not by individual image.

## Monitor the pipeline

Useful measurements include:

- Samples per class.
- Failed file count.
- Batch loading time.
- GPU idle time.
- Pixel-value range after transforms.
- Label min/max and dtype.

See the [robust image-pipeline project](../../projects/robust-image-pipeline.md) for a reusable implementation pattern.
