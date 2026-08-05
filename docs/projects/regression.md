---
title: Nonlinear regression
course: PyTorch Fundamentals
tags:
  - project
  - regression
last_reviewed: 2026-08-05
---

# Project: nonlinear regression

## Problem

Predict a curved one-dimensional function and show why a purely linear model cannot represent it.

```text
y = 0.45x³ − 0.35x + 0.2sin(4x) + noise
```

## Experiment

Compare:

```python
linear = nn.Linear(1, 1)

nonlinear = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1),
)
```

Both models use the same deterministic split, mean-squared error, optimizer family, and evaluation procedure.

## Why this matters

Stacking linear layers without nonlinear activations still produces a linear transformation. The `Tanh` activations let the second model approximate curvature.

## Run it

```bash
python examples/regression_demo.py
```

The script prints measured validation loss for both models and writes a retained comparison chart. Those values are generated locally; this page does not hard-code an accuracy claim.

## Verified local run

With seed `42` and the retained script configuration:

| Model | Validation MSE |
| --- | ---: |
| Linear | 0.40296 |
| Nonlinear network | 0.00755 |

![Scatter plot of the same nonlinear data with a linear fit and a nonlinear neural-network fit](../assets/images/regression-comparison.png)

The comparison is specific to this synthetic dataset, split, seed, architecture, and training configuration. It demonstrates representational capacity; it is not a general benchmark.

## Transferable lessons

- Visualize predictions, not only scalar loss.
- Keep the split and seed fixed during comparison.
- Match model capacity to the pattern rather than automatically adding depth.
- A successful tiny experiment can validate the complete training pipeline.
