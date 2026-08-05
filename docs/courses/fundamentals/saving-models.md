---
title: Saving and loading models
course: PyTorch Fundamentals
module: 8
tags:
  - serialization
  - reproducibility
last_reviewed: 2026-08-05
---

# Saving and loading models

## Save model parameters

```python
torch.save(model.state_dict(), "model.pth")
```

Restore them into the same architecture:

```python
model = Classifier(input_features=784, classes=26)
state = torch.load("model.pth", map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()
```

## Save a training checkpoint

```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "validation_loss": validation_loss,
    "class_names": class_names,
}, "checkpoint.pth")
```

A useful checkpoint retains enough context to resume training or interpret outputs. Architecture code, preprocessing, class order, library versions, and input shape are part of the model contract even when they are not tensors.

## Reproducibility checklist

- Record the random seed.
- Retain train/validation/test split identifiers.
- Store class order and preprocessing values.
- Record package versions.
- Save the selected validation metric.
- Verify a restored model on a known input.
- Never load an untrusted pickle-based checkpoint.

## `weights_only`

For compatible PyTorch versions, prefer safer loading modes intended for weight tensors when you do not need arbitrary serialized Python objects. Treat checkpoint files as executable-risk artifacts unless their origin is trusted.

