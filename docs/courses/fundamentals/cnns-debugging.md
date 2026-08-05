---
title: CNNs and debugging
study_context: PyTorch Fundamentals
topic_order: 6
tags:
  - cnn
  - debugging
last_reviewed: 2026-08-05
---

# CNNs, modular architectures, and debugging

## The convolutional pattern

```python
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)
```

Convolution learns local filters. ReLU introduces nonlinearity. Pooling reduces spatial resolution. Deeper blocks combine simple features into more useful patterns.

<div class="shape-tracer interactive-panel" data-shape-tracer>
  <div class="interactive-heading">CNN shape tracer</div>
  <label>Input size <input type="number" min="8" value="32" data-spatial-size></label>
  <label>Pooling blocks <input type="number" min="0" max="6" value="3" data-pool-blocks></label>
  <button type="button" data-trace-shape>Trace</button>
  <output data-shape-output aria-live="polite"></output>
</div>

## Shape formula

For one spatial dimension:

```text
output = floor((input + 2 × padding − dilation × (kernel − 1) − 1) / stride + 1)
```

For a 3 × 3 convolution with padding 1 and stride 1, height and width stay unchanged. A 2 × 2 pooling layer with stride 2 usually halves them.

## Safer classifier boundaries

Hard-coded flattened sizes are easy to break when input resolution changes. Two alternatives:

=== "Inspect the feature size"

    ```python
    with torch.no_grad():
        features = self.feature_extractor(torch.zeros(1, 3, 32, 32))
    flattened = features.numel()
    ```

=== "Use adaptive pooling"

    ```python
    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    self.classifier = nn.Linear(channels, classes)
    ```

## Debugging hooks

```python
def show_shape(name):
    def hook(module, inputs, output):
        print(name, tuple(output.shape))
    return hook

handle = model.features.register_forward_hook(show_shape("features"))
```

Remove temporary hooks after debugging with `handle.remove()`.
