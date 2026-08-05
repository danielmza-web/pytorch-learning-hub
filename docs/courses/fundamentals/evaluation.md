---
title: Evaluation and metrics
course: PyTorch Fundamentals
module: 4
tags:
  - evaluation
  - metrics
last_reviewed: 2026-08-05
---

# Validation, evaluation, and metrics

## Correct evaluation mode

```python
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
```

`model.eval()` changes dropout and batch-normalization behavior. `torch.no_grad()` reduces memory use by disabling gradient tracking.

## Choose metrics for the question

| Metric | Best used when | Important limitation |
| --- | --- | --- |
| Accuracy | Classes are reasonably balanced | Can hide minority-class failure |
| Precision | False positives are costly | Does not measure missed positives |
| Recall | False negatives are costly | Does not measure false alarms |
| F1 | Precision and recall both matter | Hides class-specific detail when averaged |
| Confusion matrix | You need error structure | Requires inspection, not one scalar |

## Avoid leakage

- Split before fitting data-dependent preprocessing.
- Never tune repeatedly against the final test set.
- Do not apply random training augmentation to validation.
- Keep class mapping identical across splits.
- Save the model selected by validation behavior, then evaluate once on test data.

## Weighted averages

If the last batch is smaller, average batch losses by sample count:

```python
running_loss += loss.item() * inputs.size(0)
epoch_loss = running_loss / len(loader.dataset)
```

Simply averaging batch averages gives the final small batch the same weight as a full batch.

