---
title: Convolution explorer
tags:
  - convolution
  - cnn
last_reviewed: 2026-08-05
---

# Convolution explorer

A convolution applies the same small kernel at many spatial positions. Reusing weights is what makes it efficient and location-aware.

![Conceptual example of a vertical-edge kernel sliding over an image and producing a feature map](../assets/images/convolution-visual.png)

This is a hand-designed filter so the response is visible. A trained CNN learns the kernel values from its own data.

<div class="kernel-lab interactive-panel" data-kernel-lab>
  <div class="interactive-heading">3 × 3 kernel explorer</div>
  <div class="kernel-controls">
    <label>Kernel
      <select data-kernel-select>
        <option value="edge">Vertical edge</option>
        <option value="horizontal">Horizontal edge</option>
        <option value="sharpen">Sharpen</option>
        <option value="blur">Box blur</option>
      </select>
    </label>
    <button type="button" data-randomize-grid>Randomize input</button>
  </div>
  <div class="kernel-stage">
    <div><strong>Input</strong><div class="matrix-grid" data-input-grid></div></div>
    <div><strong>Kernel</strong><div class="matrix-grid kernel-grid" data-kernel-grid></div></div>
    <div><strong>Valid output</strong><div class="matrix-grid output-grid" data-output-grid></div></div>
  </div>
  <output data-kernel-output aria-live="polite"></output>
</div>

## Learned filters

The example kernels above are hand-designed. In a CNN, the kernel values are parameters updated through backpropagation. Early filters often respond to edges or color transitions; deeper layers combine feature maps into task-specific patterns.

## Channels

```python
nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=3,
    padding=1,
)
```

Each of the 32 output filters spans all three RGB input channels. The result is 32 learned feature maps, not 32 colors.
