---
title: Nonlinear regression
study_context: PyTorch Fundamentals
tags:
  - project
  - regression
last_reviewed: 2026-09-27
---

# Project: nonlinear regression

One small experiment shows exactly what the activation changes: the shape of the function the model can learn.
{ .page-lead }

## The question

When does a linear layer stop being expressive enough for the pattern in the data?

## What to remember

Several linear layers still describe one linear transformation unless an activation sits between them. `Tanh` lets this small network bend its prediction around a curved target.

## Key code

```python
linear = nn.Linear(1, 1)

nonlinear = nn.Sequential(
    nn.Linear(1, 32), nn.Tanh(),
    nn.Linear(32, 32), nn.Tanh(),
    nn.Linear(32, 1),
)
```

`nn.Linear(1, 1)` can only fit a line. The hidden layers create intermediate features; each `Tanh` makes the final mapping nonlinear.

**Check yourself:** remove both `Tanh` calls. The network has more parameters, but can it follow the curve? No: its layers still combine into one affine mapping. Compare this with the [ReLU mini experiment](../guides/fundamentals/core-workflow.md#why-activations-matter): the exact activation differs, but the need for nonlinearity is the same.

## Evidence and visual

This is a reproduced result from the retained script, using its synthetic dataset, seed `42`, split, architecture, and training configuration. It is a capacity demonstration, not a general benchmark.

| Model | Validation MSE |
| --- | ---: |
| Linear | 0.40296 |
| Nonlinear network | 0.00755 |

![Reproduced chart of the same curved samples with a poor linear fit on the left and a flexible nonlinear-network fit on the right](../assets/images/regression-comparison.png)

## Interactive check

<div class="interactive-panel" data-regression-lab>
  <div class="interactive-heading">Capacity reminder</div>
  <label>Hidden width <input type="range" min="1" max="64" value="32" data-regression-width></label>
  <output data-regression-output aria-live="polite"></output>
</div>

Changing width changes the number of learned hidden features. The important switch is still the activation: width without a nonlinear activation does not make a curved function possible.

## Run it yourself

```bash
python examples/regression_demo.py
```

It prints the two validation losses and refreshes `docs/assets/images/regression-comparison.png`.

## Complete source

??? note "Open the maintained runnable script"
    ```python
    --8<-- "examples/regression_demo.py"
    ```
