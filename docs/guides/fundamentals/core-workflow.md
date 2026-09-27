---
title: Fundamentals — core workflow
study_context: PyTorch Fundamentals
tags:
  - fundamentals
  - tensors
  - training
last_reviewed: 2026-09-27
---

# Fundamentals — core workflow

Follow one path from values in memory to a model you can evaluate. Read it in order once; later, jump straight to the step you need to debug.
{ .page-lead }

## What this guide connects

```text
tensor → Dataset → DataLoader → model → output → loss
                                      ↓
                         gradients → optimizer update
                                      ↓
                              validation evidence
```

By the end, you should be able to locate where a batch is created, explain the model's input and output shapes, identify where parameters change, and separate training from evaluation. In classification, the output is often logits; regression usually produces predicted values.

**How to study this page:** for each section, run the tiny example, predict the output shape or value, then check it. The goal is to explain *why* each line exists before memorizing its spelling.

| Section | Main question | Useful next example |
| --- | --- | --- |
| [Tensors, shapes, dtype, and device](#tensors-shapes-dtype-and-device) | What does the data mean before it enters a model? | [Nonlinear regression](../../projects/regression.md) |
| [Dataset, transforms, and DataLoader](#dataset-transforms-and-dataloader) | How do individual samples become batches? | [Robust image pipeline](../../projects/robust-image-pipeline.md) |
| [Models, activations, and logits](#models-activations-and-logits) | What contract does an `nn.Module` provide? | [EMNIST classifier](../../projects/emnist.md) |
| [Loss, autograd, and optimizers](#loss-autograd-and-optimizers) | How does an error become a weight update? | [Training loop](#the-complete-training-loop) |
| [Validation and metrics](#validation-and-metrics) | How do we check behavior without learning from the check? | [Vision and real data](vision-real-data.md) |

## Tensors, shapes, dtype, and device

A tensor is a multidimensional array. Its values are only part of its meaning: **shape**, **dtype**, and **device** are also part of the program.

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)   # torch.Size([2, 2])
print(x.dtype)   # torch.float32
print(x.device)  # cpu
```

### Read the shape before the layer

| Data | Typical shape | Meaning |
| --- | --- | --- |
| Tabular batch | `[N, F]` | samples, features |
| Image batch | `[N, C, H, W]` | samples, channels, height, width |
| Sequence batch | `[N, L, E]` | samples, sequence length, embedding size |
| Class logits | `[N, K]` | samples, classes |
| Segmentation logits | `[N, K, H, W]` | samples, classes, spatial prediction |

For example, `[32, 3, 224, 224]` means 32 RGB images at 224 × 224 pixels. The first dimension is the batch; it must survive when features are flattened.

<div class="tensor-lab interactive-panel" data-tensor-lab>
  <div class="interactive-heading">Image tensor reader</div>
  <label>Batch <input type="number" min="1" value="32" data-dim="batch"></label>
  <label>Channels <input type="number" min="1" value="3" data-dim="channels"></label>
  <label>Height <input type="number" min="1" value="224" data-dim="height"></label>
  <label>Width <input type="number" min="1" value="224" data-dim="width"></label>
  <output data-tensor-output aria-live="polite"></output>
</div>

### Reshape without changing the data

```python
x = torch.arange(12).reshape(3, 4)

x.unsqueeze(0).shape       # [1, 3, 4]
x.transpose(0, 1).shape    # [4, 3]
torch.flatten(x).shape      # [12]
torch.flatten(x, 1).shape   # preserves dimension 0 when x is batched
```

`reshape` changes how compatible values are viewed; it does not add or remove them. `torch.flatten(x, start_dim=1)` preserves the batch boundary.

For a batch of 16 grayscale 28 × 28 images, a dense classifier needs 784 features per image:

```python
from torch import nn

images = torch.zeros(16, 1, 28, 28)   # [batch, channels, height, width]
features = torch.flatten(images, 1)    # [16, 784]; keep the batch axis
scores = nn.Linear(784, 10)(features)   # [16, 10]; one score per class
```

**Check yourself:** why would `torch.flatten(images)` be wrong here? It would merge all 16 images into one vector and lose which sample each label belongs to.

`nn.Linear(in_features, out_features)` expects the last input dimension to equal `in_features`:

```text
[N, in_features] × [in_features, out_features] → [N, out_features]
```

If PyTorch reports `mat1 and mat2 shapes cannot be multiplied`, print the tensor immediately before the linear layer.

### Broadcasting

Broadcasting combines compatible shapes without manually copying data. Dimensions are compared from right to left; each pair must be equal or one must be `1`.

```python
batch = torch.ones(4, 3)
bias = torch.tensor([0.1, 0.2, 0.3])
result = batch + bias  # [4, 3]
```

<div class="broadcast-lab interactive-panel" data-broadcast-lab>
  <div class="interactive-heading">Broadcasting checker</div>
  <label>Shape A <input value="4,3" data-shape-a aria-label="First tensor shape"></label>
  <label>Shape B <input value="3" data-shape-b aria-label="Second tensor shape"></label>
  <button type="button" data-check-broadcast>Check shapes</button>
  <output data-broadcast-output aria-live="polite"></output>
</div>

### Dtype and device

Neural-network inputs are normally floating point; single-label class targets for `CrossEntropyLoss` are normally `torch.long`.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = inputs.to(device=device, dtype=torch.float32)
labels = labels.to(device=device, dtype=torch.long)
```

The model, inputs, targets, and helper tensors used in the same operation must share a device.

**Next:** see shape, device, autograd, and nonlinearity together in [nonlinear regression](../../projects/regression.md).

## Dataset, transforms, and DataLoader

![Conceptual path from image file through transforms and Dataset to a DataLoader batch tensor](../../assets/images/data-pipeline-visual.png)

The diagram is structural, not a measured experiment. It shows where a file becomes a tensor and where samples become a batch.

| Part | Owns | Produces |
| --- | --- | --- |
| `Dataset` | locating and preparing one valid example | `(input, label)` |
| transforms | deterministic preparation or training-only augmentation | a model-ready tensor |
| `DataLoader` | batching, order, parallel loading | batches of examples |
| training loop | device transfer and model updates | loss history and changed parameters |

### Dataset: one sample

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = load_image(self.paths[index])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]
```

Load lazily in `__getitem__`; most image projects should not put the complete dataset in memory.

### Transforms: prepare consistently

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

Random, label-preserving augmentation belongs in training only. Validation must be stable between epochs. `ToTensor()` changes layout to `[C, H, W]` and normally scales byte pixels to `0–1`; normalization is a separate operation.

`mean` and `std` above must come from training data (or a documented pretrained model's expected preprocessing). Use the **same** fixed normalization for validation; fitting it on validation would leak information into training decisions. A horizontal flip is safe only when it preserves the label for your task: it can change the meaning of some symbols or medical images.

### DataLoader: samples become batches

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)
```

<div class="batch-lab interactive-panel" data-batch-lab>
  <div class="interactive-heading">Batch calculator</div>
  <label>Samples <input type="number" min="1" value="2100" data-samples></label>
  <label>Batch size <input type="number" min="1" value="32" data-batch-size></label>
  <label class="check-row"><input type="checkbox" data-drop-last> Drop incomplete batch</label>
  <output data-batch-output aria-live="polite"></output>
</div>

Inspect one batch before defining the model:

```python
images, labels = next(iter(train_loader))
print(images.shape, images.dtype, images.min(), images.max())
print(labels.shape, labels.dtype, labels.min(), labels.max())
```

This catches a surprising shape, incorrect label dtype, or wrong class range before a long run.

**Next:** use the [robust image-pipeline project](../../projects/robust-image-pipeline.md) to see validation, corrupt-file reporting, batching, and artifacts together.

## Models, activations, and logits

An `nn.Module` owns trainable layers and defines how an input becomes an output.

```python
from torch import nn

class Classifier(nn.Module):
    def __init__(self, input_features: int, classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        return self.network(x)  # [batch, classes]
```

- `__init__` creates reusable layers and registers their parameters.
- `forward` describes the transformation.
- `model(x)` is the normal call; it preserves hooks and PyTorch internals.
- The output is raw **logits**, one score per class.

### Why activations matter

```python
linear = nn.Linear(1, 1)

nonlinear = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 1),
)
```

Try the activation by itself:

```python
values = torch.tensor([-2.0, 0.0, 3.0])
print(nn.ReLU()(values))  # tensor([0., 0., 3.])
```

`ReLU(x) = max(0, x)`: it replaces a **negative intermediate input** with zero and leaves a positive input unchanged. It does *not* make values negative. A later linear output layer can still produce a negative prediction. Its zero side and sloped side let different hidden units switch on in different regions.

```python
model = nn.Sequential(
    nn.Linear(1, 3),  # one distance → three learned hidden values
    nn.ReLU(),        # zero negative hidden values; create a bend
    nn.Linear(3, 1),  # combine them into one prediction
)
```

In the course's delivery-time lab, a single line could not follow a curved relationship; adding hidden units and an activation let the model express bends. If you remove `ReLU`, consecutive `Linear` layers collapse to one affine map, however many you stack. `Tanh` is another nonlinear choice, with a smooth output between −1 and 1. **Check yourself:** in this model, which layer can make the final prediction negative? The last `Linear` layer.

![Measured comparison from the retained regression script: a line misses a curved pattern while the nonlinear network follows it](../../assets/images/regression-comparison.png)

This chart is reproduced by the retained regression script with a fixed seed. It demonstrates model capacity, not a general benchmark.

### Logits, probabilities, and predictions

For single-label multiclass classification:

```python
logits = model(inputs)                 # [batch, classes]
loss = nn.CrossEntropyLoss()(logits, labels)
predictions = logits.argmax(dim=1)    # [batch]
probabilities = logits.softmax(dim=1) # only when probabilities are needed
```

Do not apply softmax before `CrossEntropyLoss`; the loss already combines the stable operations it needs.

**Next:** the [EMNIST project](../../projects/emnist.md) compares a dense classifier with a CNN and produces predictions, curves, metrics, and a checkpoint.

## Loss, autograd, and optimizers

![Conceptual update cycle from batch to model, logits, loss, gradients, and optimizer step](../../assets/images/training-cycle-visual.png)

Loss turns model behavior into one differentiable scalar. Autograd follows the operations that produced it, and the optimizer uses the resulting gradients to update parameters.

```python
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = (w * x) ** 2
y.backward()
print(w.grad)
```

Gradients accumulate by default. That is why each independent update clears old gradients before backpropagation.

### The complete training loop

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
def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    samples_seen = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        samples_seen += inputs.size(0)

    return total_loss / samples_seen
```

| Step | Reads | Changes |
| --- | --- | --- |
| `zero_grad()` | optimizer parameter list | clears stored gradients |
| `model(inputs)` | inputs and parameters | creates activations and the computation graph |
| `loss_fn(...)` | logits and targets | creates the scalar objective |
| `loss.backward()` | computation graph | accumulates parameter gradients |
| `optimizer.step()` | parameters and gradients | updates parameters |

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

A learning rate that is too high can make loss unstable; one that is too low can make progress impractically slow. Before a full run, try to overfit one small batch. If loss cannot fall, inspect labels, loss pairing, gradients, and optimizer ownership.

Advanced variations keep this basic order: gradient clipping goes after `backward()` and before `step()`; accumulation delays `step()`; mixed precision wraps the forward/loss work and scales gradients.

## Validation and metrics

Evaluation uses the same model without updating its parameters.

```python
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total
```

`model.eval()` changes dropout and batch-normalization behavior. `torch.no_grad()` disables gradient tracking and reduces memory use. Use both; neither computes a metric by itself.

Notice what evaluation leaves out: there is no `zero_grad()`, `backward()`, or `step()`. Validation measures the current weights; it must not train on the answers it is supposed to check.

### Choose evidence for the question

| Metric | Useful when | Limitation |
| --- | --- | --- |
| Accuracy | classes are reasonably balanced | can hide minority-class failure |
| Precision | false positives are costly | ignores missed positives |
| Recall | false negatives are costly | ignores false alarms |
| F1 | precision and recall both matter | averages can hide class-specific failure |
| Confusion matrix | you need the structure of mistakes | requires inspection, not one scalar |

Weight epoch loss by sample count so a smaller final batch does not count like a full batch. Divide by the number of samples actually processed if `drop_last=True` or a custom sampler skips examples:

```python
running_loss += loss.item() * inputs.size(0)
epoch_loss = running_loss / samples_seen
```

### Prevent leakage

- Split before fitting data-dependent preprocessing.
- Keep class-to-index mapping identical across splits.
- Do not apply random training augmentation to validation.
- Tune with validation, not repeatedly with the final test set.
- Select a checkpoint using validation behavior; evaluate on test once.

## Core-workflow checklist

Before trusting a run, answer these questions:

1. What does each input dimension mean?
2. Are input, target, and parameters on the same device?
3. Does the Dataset return one valid sample and label?
4. Do model outputs have shape `[batch, classes]`?
5. Does the loss match the output and target representation?
6. Are gradients cleared, calculated, and applied in the intended order?
7. Does evaluation use stable data, `eval()`, and no gradients?
8. Can the pipeline overfit one tiny batch?

Continue with [Fundamentals — vision and real data](vision-real-data.md), where the same workflow is applied to images, CNNs, imperfect files, and generalization.
