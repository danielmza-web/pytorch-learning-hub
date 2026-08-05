---
title: Overfitting and regularization
course: PyTorch Fundamentals
module: 7
tags:
  - overfitting
  - regularization
last_reviewed: 2026-08-05
---

# Overfitting and regularization

Overfitting is a gap between fitting the training examples and generalizing to unseen examples.

<div class="curve-lab interactive-panel" data-curve-lab>
  <div class="interactive-heading">Learning-curve comparison</div>
  <label>Regularization strength <input type="range" min="0" max="100" value="35" data-regularization></label>
  <canvas width="720" height="260" data-curve-canvas aria-label="Illustrative training and validation loss curves"></canvas>
  <output data-curve-output aria-live="polite"></output>
  <small>Illustrative curves—not benchmark results.</small>
</div>

## Diagnose before changing the model

| Observation | Likely issue | First checks |
| --- | --- | --- |
| Train and validation both poor | Underfitting or broken pipeline | Labels, loss, learning rate, capacity |
| Train improves; validation worsens | Overfitting | Split quality, augmentation, regularization |
| Loss becomes NaN | Numerical instability | Learning rate, invalid input, exploding gradients |
| Accuracy is high but one class fails | Imbalance | Per-class recall and confusion matrix |

## Regularization tools

### Data augmentation

Create label-preserving variation. Augmentation should reflect realistic variation; an upside-down vehicle or distorted character may not preserve the task.

### Weight decay

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
```

### Dropout

```python
self.dropout = nn.Dropout(0.3)
```

Dropout is active during `model.train()` and disabled during `model.eval()`.

### Early stopping

Keep the state associated with the best validation metric rather than automatically using the final epoch.

!!! tip "Change one hypothesis at a time"
    Compare experiments with the same split, seed, metrics, and evaluation procedure. Otherwise an apparent improvement may come from a changed test rather than a better model.

