---
title: Tensors and autograd
study_context: PyTorch Fundamentals
topic_order: 1
tags:
  - tensors
  - autograd
last_reviewed: 2026-08-05
---

# Tensors and autograd

A tensor is a multidimensional array with a shape, dtype, and device. Those three properties explain most early PyTorch errors.

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)   # torch.Size([2, 2])
print(x.dtype)   # torch.float32
print(x.device)  # cpu
```

## Shape operations

```python
x = torch.arange(12)
x = x.reshape(3, 4)

x.unsqueeze(0).shape  # [1, 3, 4]
x.flatten().shape      # [12]
x.transpose(0, 1).shape  # [4, 3]
```

`reshape` changes the view of the values; it does not invent or remove values. The product of the dimensions must remain compatible.

## Broadcasting

Broadcasting lets PyTorch combine compatible shapes without manually copying data.

```python
batch = torch.ones(4, 3)
bias = torch.tensor([0.1, 0.2, 0.3])
result = batch + bias  # bias is applied to each row
```

Dimensions are compared from right to left. Each pair must be equal, or one of them must be `1`.

<div class="broadcast-lab interactive-panel" data-broadcast-lab>
  <div class="interactive-heading">Broadcasting checker</div>
  <label>Shape A <input value="4,3" data-shape-a aria-label="First tensor shape"></label>
  <label>Shape B <input value="3" data-shape-b aria-label="Second tensor shape"></label>
  <button type="button" data-check-broadcast>Check shapes</button>
  <output data-broadcast-output aria-live="polite"></output>
</div>

## Gradients

```python
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = (w * x) ** 2
y.backward()
print(w.grad)  # dy/dw
```

Autograd records operations that involve tensors requiring gradients. Calling `backward()` applies the chain rule and accumulates gradients in leaf tensors.

!!! warning "Gradients accumulate"
    Repeated calls to `backward()` add to existing gradients. Optimizers therefore call `optimizer.zero_grad()` before each new update.

## Moving tensors

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
```

Operations require participating tensors to be on the same device. The [common errors guide](../../reference/common-errors.md) includes a device checklist.

## Self-check

??? question "Why is `[32, 3, 224, 224]` not one image?"
    The first dimension is the batch. It contains 32 images; each image has three channels and a 224 × 224 spatial grid.

??? question "Can `[8, 1, 10]` broadcast with `[7, 10]`?"
    No. From the right, `10` matches `10`, but `1` can expand to `7`; the remaining `8` has no matching leading dimension only if the second shape is treated as `[1, 7, 10]`, so it actually can broadcast to `[8, 7, 10]`. Writing the implicit leading `1` makes the rule easier to see.
