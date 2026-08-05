---
title: PyTorch cheatsheet
tags:
  - cheatsheet
last_reviewed: 2026-08-05
---

# PyTorch cheatsheet

## Inspect

```python
print(x.shape, x.dtype, x.device)
print(x.min().item(), x.max().item())
print(labels.unique())
print(model)
```

## Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs, targets = inputs.to(device), targets.to(device)
```

## Train

```python
model.train()
optimizer.zero_grad()
logits = model(inputs)
loss = loss_fn(logits, targets)
loss.backward()
optimizer.step()
```

## Evaluate

```python
model.eval()
with torch.no_grad():
    logits = model(inputs)
    predictions = logits.argmax(dim=1)
```

## Save and restore

```python
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```

## Count parameters

```python
trainable = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
```

## Reproducibility seed

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

