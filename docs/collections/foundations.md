---
title: First models and tensors
tags:
  - collection
  - foundations
last_reviewed: 2026-08-06
---

# First models and tensors

This collection answers the questions that make the rest of PyTorch legible: what a tensor represents, what a model transforms, and how parameters change from examples.

## The mental model

```text
numbers + shape + dtype + device
             ↓
        model(inputs)
             ↓
          predictions
             ↓
        loss(predictions, targets)
             ↓
          gradients → optimizer update
```

A tensor is not only a container of values. Its shape tells a model how to interpret those values, its dtype controls arithmetic, and its device says where the work happens. When something fails, inspect these three facts before changing an architecture.

## Questions to open

| If you are asking… | Open this |
| --- | --- |
| Why does `[32, 3, 224, 224]` matter? | [Tensor shapes](../concepts/tensor-shapes.md) |
| Why cannot a line fit a curve? | [Nonlinear regression](../projects/regression.md) |
| How does broadcasting decide whether an operation is valid? | [Tensors and autograd](../courses/fundamentals/tensors-autograd.md) |
| What does `loss.backward()` actually prepare? | [Training loop](../concepts/training-loop.md) |

## A reliable first experiment

The regression project is intentionally small because it exposes the entire loop without a dataset download: construct data, define a linear baseline, add nonlinear activations, optimize mean-squared error, compare predictions, and save a chart. A tiny project is enough to learn the order of operations when each part is visible.

```python
optimizer.zero_grad(set_to_none=True)
prediction = model(inputs)
loss = loss_fn(prediction, targets)
loss.backward()
optimizer.step()
```

The update order never becomes optional. More advanced projects change the model and data, but keep this contract.

## Keep nearby

- A model normally returns raw values called logits; the loss function decides how to interpret them.
- Gradients accumulate by default, so clearing them is part of every training batch.
- Nonlinear activations change what a network can represent; adding only linear layers does not.
- `model.to(device)` and every input tensor must agree on the same device.
