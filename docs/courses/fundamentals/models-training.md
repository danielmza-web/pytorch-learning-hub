---
title: Models and training
study_context: PyTorch Fundamentals
topic_order: 3
tags:
  - nn-module
  - training
  - optimization
last_reviewed: 2026-08-05
---

# Models, loss, optimizers, and training

## Define a model

```python
from torch import nn

class Classifier(nn.Module):
    def __init__(self, input_features, classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        return self.network(x)
```

`__init__` defines reusable layers. `forward` defines the data path. Calling `model(x)` invokes `forward` while preserving PyTorch's hooks and internal behavior.

## Logits and loss

For single-label multiclass classification:

```python
loss_fn = nn.CrossEntropyLoss()
logits = model(inputs)
loss = loss_fn(logits, labels)
```

The model returns raw logits. `CrossEntropyLoss` combines the necessary log-softmax and negative log-likelihood operations.

## The update sequence

<div class="training-stepper interactive-panel" data-training-stepper>
  <div class="interactive-heading">Training-loop stepper</div>
  <ol>
    <li data-train-step="0">Clear old gradients</li>
    <li data-train-step="1">Run the forward pass</li>
    <li data-train-step="2">Measure the loss</li>
    <li data-train-step="3">Backpropagate gradients</li>
    <li data-train-step="4">Update parameters</li>
  </ol>
  <div class="stepper-actions">
    <button type="button" data-step-prev>Previous</button>
    <button type="button" data-step-next>Next</button>
  </div>
  <output data-step-output aria-live="polite"></output>
</div>

```python
model.train()

for inputs, labels in train_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits = model(inputs)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
```

## Optimizers

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

The optimizer owns references to trainable parameters and applies updates using their gradients. A learning rate that is too large can make loss unstable; one that is too small can make meaningful progress impractically slow.

## Verify learning with one batch

Before a full run, repeatedly train on one small batch. A sufficiently expressive model should drive its loss down. If it cannot, inspect the pipeline, labels, loss choice, and optimizer before scaling up.
