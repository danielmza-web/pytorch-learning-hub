---
title: CNNs for images
tags:
  - start
  - cnn
  - convolution
last_reviewed: 2026-08-06
---

# 3. CNNs for images

A dense layer sees one long list of pixels. A convolutional neural network keeps pixels arranged as an image while it learns small visual patterns that can appear anywhere.

## A filter produces a feature map

![Conceptual convolution: a 3 by 3 vertical-edge kernel slides over an input image and produces strong responses at the edge](../assets/images/convolution-visual.png)

The pictured kernel is hand-designed so its effect is easy to see. In a real CNN, kernels are trainable weights. Early layers often respond to edges or simple texture; later layers combine those responses into more useful features.

```python
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
```

This reads three RGB channels and creates 32 learned feature maps. Those output channels are pattern detectors, not 32 colours.

## Follow the shape through the model

![Conceptual CNN path showing batch shape after convolution, pooling, and classification](../assets/images/cnn-shape-visual.png)

- A padded `3 × 3` convolution with stride `1` normally preserves height and width.
- `MaxPool2d(2)` normally halves height and width.
- The classifier finally turns features into `[batch, classes]` logits.

Use adaptive pooling when possible so a changed image size does not break a hard-coded flatten dimension.

```python
self.features = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d((1, 1)),
)
self.classifier = nn.Linear(64, classes)
```

Try the [interactive convolution explorer](../concepts/convolution.md), then open the [Nature CNN project](../projects/nature-cnn.md) to run the whole pipeline.
