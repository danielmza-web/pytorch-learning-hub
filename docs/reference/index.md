---
title: PyTorch reference and troubleshooting
tags:
  - reference
  - cheatsheet
  - debugging
last_reviewed: 2026-08-06
---

# PyTorch reference and troubleshooting

Use this page when you remember the workflow but need a dependable pattern, a debugging order, or a definition. For connected explanations, return to [Fundamentals — core workflow](../guides/fundamentals/core-workflow.md) or [vision and real data](../guides/fundamentals/vision-real-data.md).

## Short reminder

1. Shapes are part of the program: image batches normally use `[N, C, H, W]`.
2. Model, inputs, targets, and helper tensors used together share a device.
3. A Dataset returns one sample; a DataLoader returns a batch.
4. A classifier normally returns raw logits.
5. The update order is clear → predict → measure → differentiate → update.
6. Evaluation uses both `model.eval()` and disabled gradients.
7. Convolution grows useful feature channels; pooling or stride reduces spatial size.
8. Training behavior matters only when compared with validation and real-use data.
9. Save weights together with the context needed to interpret them.

## Cheatsheet

### Inspect data and models

```python
print(x.shape, x.dtype, x.device)
print(x.min().item(), x.max().item())
print(labels.unique())
print(model)
```

### Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs, targets = inputs.to(device), targets.to(device)
```

### Train

```python
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)
loss = loss_fn(logits, targets)
loss.backward()
optimizer.step()
```

### Evaluate

```python
model.eval()
with torch.no_grad():
    logits = model(inputs)
    predictions = logits.argmax(dim=1)
```

### Save and restore

```python
torch.save(model.state_dict(), "model.pth")
state = torch.load("model.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
```

### Count trainable parameters

```python
trainable = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
```

### Fix common random seeds

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

## Common errors

### `mat1 and mat2 shapes cannot be multiplied`

The final input dimension does not match `nn.Linear(in_features, ...)`.

```python
print(x.shape)  # immediately before the linear layer
```

Preserve the batch when flattening: `torch.flatten(x, start_dim=1)`.

### `Expected all tensors to be on the same device`

Move the model, inputs, targets, and newly created helper tensors to the same device. Inspect `.device` at the failing operation.

### `Target N is out of bounds`

For `K` output classes, labels for `CrossEntropyLoss` must normally be integer ids in `0..K-1`. Check one-based source labels and the class mapping.

### Loss does not improve

1. Inspect input values, labels, shapes, and dtypes.
2. Verify the output/loss pairing.
3. Try to overfit one small batch.
4. Confirm parameters receive gradients.
5. Confirm the optimizer owns those parameters.
6. Inspect the learning rate.

### Validation changes unexpectedly

- Call `model.eval()` and disable gradients.
- Remove random validation transforms.
- Use a fixed validation split.
- Keep class mapping and deterministic preprocessing consistent.
- Check for overlap or entity leakage between splits.

### CUDA out of memory

- Reduce batch size first.
- Do not retain computation graphs in Python lists.
- Store `loss.item()` rather than the loss tensor.
- Evaluate inside `torch.no_grad()`.
- Reduce input resolution or model size after measuring the bottleneck.

### NaN loss

- Inspect inputs for NaN or infinity.
- Reduce the learning rate.
- Check logarithms, divisions, and normalization.
- Confirm target dtypes and ranges.
- Clip gradients only when they genuinely explode.

## Debug in this order

1. Print shapes, dtypes, devices, and value ranges.
2. Check labels and class range.
3. Verify model output shape and loss pairing.
4. Confirm train/eval mode.
5. Overfit one tiny batch.
6. Inspect learning curves and class-specific errors.
7. Only then change architecture or regularization.

## Glossary

**Activation**
: The output produced by a layer or nonlinear function.

**Autograd**
: PyTorch's automatic differentiation system.

**Batch**
: A group of samples processed together before one optimizer update.

**Epoch**
: One pass through the training dataset.

**Feature map**
: One channel of activations produced by a convolutional filter.

**Gradient**
: The derivative of the objective with respect to a parameter.

**Logit**
: A raw model score before conversion to a probability.

**Loss**
: A differentiable scalar objective measuring prediction error.

**Parameter**
: A trainable tensor such as a weight or bias.

**Regularization**
: A technique intended to improve generalization rather than only training fit.

**Tensor**
: A multidimensional array carrying shape, dtype, and device information.
