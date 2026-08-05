---
title: CNNs and model inspection
tags:
  - collection
  - cnn
last_reviewed: 2026-08-06
---

# CNNs and model inspection

CNNs make image models practical by applying local filters across an image, gradually trading spatial detail for learned feature channels. This collection focuses on reading that transformation and checking whether it generalizes.

## Read the shape story

```text
[batch, 3, 32, 32]
  → convolution: channels grow
  → pooling/stride: width and height shrink
  → classifier: [batch, classes]
```

Every layer has an input and output contract. Print or hook shapes at architecture boundaries. Most CNN failures are easier to find at the first unexpected shape than at the final loss error.

## Questions to open

| If you are asking… | Open this |
| --- | --- |
| What does a convolution filter do? | [Convolution explorer](../concepts/convolution.md) |
| How do I trace CNN dimensions? | [CNNs and debugging](../courses/fundamentals/cnns-debugging.md) |
| Is the model learning or memorizing? | [Generalization](../concepts/generalization.md) |
| How do I preserve the best usable model? | [Saving models](../courses/fundamentals/saving-models.md) |
| How does this look with real data and artifacts? | [Nature CNN project](../projects/nature-cnn.md) |

## End-to-end reference project

The Nature CNN project downloads CIFAR-100 through TorchVision, selects 15 nature classes, trains a regularized CNN, then saves predictions, a confusion matrix, learning curves, metrics, and a checkpoint. The default subset is deliberately small for CPU. The optional full GPU mode uses the full selected dataset and a longer epoch budget.

## Keep nearby

- A train-loss decrease is not enough; compare it with validation behavior.
- Dropout, weight decay, augmentation, and model size solve different problems and should be changed deliberately.
- Save `state_dict`, class names, preprocessing, and run history together.
