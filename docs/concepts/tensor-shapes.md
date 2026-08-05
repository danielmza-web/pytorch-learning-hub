---
title: Tensor shapes
tags:
  - tensors
  - shapes
last_reviewed: 2026-08-05
---

# Tensor shapes

## Common conventions

| Data | Typical shape | Meaning |
| --- | --- | --- |
| Tabular batch | `[N, F]` | samples, features |
| Image batch | `[N, C, H, W]` | samples, channels, height, width |
| Sequence batch | `[N, L, E]` | samples, sequence length, embedding size |
| Class logits | `[N, K]` | samples, classes |
| Segmentation logits | `[N, K, H, W]` | samples, classes, spatial prediction |

## Interactive shape reader

<div class="tensor-lab interactive-panel" data-tensor-lab>
  <div class="interactive-heading">Image tensor reader</div>
  <label>Batch <input type="number" min="1" value="32" data-dim="batch"></label>
  <label>Channels <input type="number" min="1" value="3" data-dim="channels"></label>
  <label>Height <input type="number" min="1" value="224" data-dim="height"></label>
  <label>Width <input type="number" min="1" value="224" data-dim="width"></label>
  <output data-tensor-output aria-live="polite"></output>
</div>

## Flatten carefully

```python
x = torch.flatten(x, start_dim=1)
```

`start_dim=1` preserves the batch dimension. Flattening from dimension 0 would combine samples with their features and destroy the batch boundary.

## Matrix multiplication

`nn.Linear(in_features, out_features)` expects the final input dimension to equal `in_features`:

```text
[N, in_features] × [in_features, out_features] → [N, out_features]
```

When PyTorch reports that `mat1 and mat2 shapes cannot be multiplied`, inspect the tensor immediately before the linear layer.

