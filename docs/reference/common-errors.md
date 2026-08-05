---
title: Common errors
tags:
  - debugging
  - errors
last_reviewed: 2026-08-05
---

# Common errors

## `mat1 and mat2 shapes cannot be multiplied`

The final feature dimension does not match `nn.Linear(in_features, ...)`.

```python
print(x.shape)  # immediately before the linear layer
```

## `Expected all tensors to be on the same device`

Move the model, inputs, targets, and newly created helper tensors to the same device.

## `Target N is out of bounds`

For `K` output classes, labels must normally be integer ids in `0..K-1`. Check one-based dataset labels and class mapping.

## Loss does not improve

1. Verify labels and loss pairing.
2. Try to overfit one small batch.
3. Check that parameters have gradients.
4. Check optimizer ownership.
5. Inspect the learning rate.
6. Confirm inputs contain meaningful variation.

## Validation changes unexpectedly

- Call `model.eval()`.
- Disable gradients.
- Remove random validation transforms.
- Use a fixed validation split.
- Check that preprocessing matches training where it should.

## CUDA out of memory

- Reduce batch size.
- Avoid retaining computation graphs in lists.
- Use `loss.item()` when storing scalar history.
- Evaluate inside `torch.no_grad()`.
- Reduce input resolution or model size only after measuring the bottleneck.

## NaN loss

- Inspect inputs for NaN or infinity.
- Reduce the learning rate.
- Check logarithms, divisions, and normalization.
- Consider gradient clipping when gradients genuinely explode.
- Confirm target types and ranges.

