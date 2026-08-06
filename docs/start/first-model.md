---
title: Tensors and a first model
tags:
  - start
  - tensors
  - linear
last_reviewed: 2026-08-06
---

# 1. Tensors and a first model

Before a model can learn, data needs a shape the model understands. A tensor is PyTorch's container for that data: a number, list, table, image, or batch of images.

## Read shape before reading the model

```python
images.shape  # [32, 3, 224, 224]
```

This means **32 images**, **3 colour channels**, and **224 × 224 pixels**. The first dimension is normally the batch and must not be flattened away.

```python
x = torch.tensor([25.0])       # [1]
x = x.unsqueeze(0)             # [1, 1] — add a batch-like dimension
x = x.squeeze(0)               # [1]    — remove that size-one dimension
```

Open the [shape explorer](../concepts/tensor-shapes.md) whenever the dimensions are unclear.

## Linear is a baseline, not a universal pattern matcher

```python
linear = nn.Linear(1, 1)

nonlinear = nn.Sequential(
    nn.Linear(1, 32), nn.Tanh(),
    nn.Linear(32, 1),
)
```

`nn.Linear` learns a weighted combination of its inputs. Stacking only linear layers is still one linear transformation. An activation such as `ReLU` or `Tanh` is what lets the network model a bend or decision boundary.

![Measured comparison from the retained regression script: a line misses a curved pattern while the nonlinear network follows it](../assets/images/regression-comparison.png)

## The smallest useful model contract

```python
class Classifier(nn.Module):
    def __init__(self, input_features: int, classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        return self.layers(x)  # logits shaped [batch, classes]
```

- `__init__` creates reusable layers and parameters.
- `forward` describes the transformation from input to output.
- `model(x)` is the normal way to call `forward`.
- The final output is **logits**, not probabilities.

Next: [build a complete classifier](first-classifier.md).
