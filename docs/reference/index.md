---
title: PyTorch reference and troubleshooting
tags:
  - reference
  - cheatsheet
  - debugging
last_reviewed: 2026-09-27
---

# PyTorch reference and troubleshooting

Use this page when you know the goal but need the next command or check. For the full reasoning, return to the [core workflow](../guides/fundamentals/core-workflow.md) or [vision guide](../guides/fundamentals/vision-real-data.md).
{ .page-lead }

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

## Choosing the next tool

The course's later image-model work adds choices to the same training loop. Use this as a prompt to investigate, not as a replacement for a controlled experiment.

| When you need to… | First choice to consider | Check before trusting it |
| --- | --- | --- |
| compare classifiers with uneven classes | precision, recall, F1, and a confusion matrix alongside accuracy | per-class results and the cost of each error type |
| change how fast a model learns | tune the optimizer's learning rate, then consider a scheduler | validation behavior at the same data split and run budget |
| search several settings | record trials and their validation objective, whether manual or automated | that the test set stays outside the search |
| reuse an image model | use the weights' recommended preprocessing and replace the classifier head | class order, input shape, and which parameters are trainable |
| choose between models | compare quality with latency, memory, and parameter count | measurements on the intended device |

These topics correspond to the tuning, efficiency, TorchVision, and transfer-learning labs already organized in the second course of the [PyTorch for Deep Learning certificate](https://www.coursera.org/specializations/pytorch-for-deep-learning). The notes below are a compact reminder while that course is still in progress.

### Course 2: what changes in practice

**Metrics:** accuracy counts all correct predictions, but can hide a weak minority class. Read the confusion matrix first, then choose precision when false alarms matter or recall when missed positives matter. Keep the final test set outside tuning.

**Learning rate:** the optimizer changes weights; a scheduler changes the optimizer's learning rate over time. For an epoch-based `StepLR`, call `scheduler.step()` after that epoch's training updates:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer)  # your training function
    scheduler.step()  # after five calls, multiply the learning rate by 0.1
```

**Transfer learning:** reuse learned visual features and train a new final classifier first. The replacement layer must have one output per class; the optimizer should receive only parameters you intend to train:

```python
from torchvision.models import resnet18, ResNet18_Weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
for parameter in model.parameters():
    parameter.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
preprocess = weights.transforms()  # use for the model's expected input format
```

The frozen backbone still participates in the forward pass; `requires_grad=False` prevents its weights from being updated. Decide on fine-tuning only after checking validation behavior. If you change which layers are trainable, rebuild the optimizer for the new parameter set.

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
