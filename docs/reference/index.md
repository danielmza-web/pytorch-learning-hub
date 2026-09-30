---
title: PyTorch reference and troubleshooting
tags:
  - reference
  - cheatsheet
  - debugging
last_reviewed: 2026-09-30
---

# PyTorch reference and troubleshooting

Use this page when you know the goal but need the next command or check. For the full reasoning, return to the [core workflow](../guides/fundamentals/core-workflow.md) or [vision guide](../guides/fundamentals/vision-real-data.md).
{ .page-lead }

## Short reminder

1. Shapes are part of the program: image batches normally use `[N, C, H, W]`.
2. Model, inputs, targets, and helper tensors used together share a device.
3. A Dataset returns one sample; a DataLoader returns a batch.
4. A classifier normally returns raw logits.
5. The update order is clear → predict → measure → differentiate → update.
6. Evaluation uses both `model.eval()` and disabled gradients.
7. Convolution grows useful feature channels; pooling or stride reduces spatial size.
8. Training behavior matters only when compared with validation and real-use data.
9. Save weights together with the context needed to interpret them.

## Choosing the next tool

The course's later image-model work adds choices to the same training loop. Use this as a prompt to investigate, not as a replacement for a controlled experiment.

| When you need to… | First choice to consider | Check before trusting it |
| --- | --- | --- |
| compare classifiers with uneven classes | precision, recall, F1, and a confusion matrix alongside accuracy | per-class results and the cost of each error type |
| change how fast a model learns | tune the optimizer's learning rate, then consider a scheduler | validation behavior at the same data split and run budget |
| search several settings | record trials and their validation objective, whether manual or automated | that the test set stays outside the search |
| reuse an image model | use the weights' recommended preprocessing and replace the classifier head | class order, input shape, and which parameters are trainable |
| choose between models | compare quality with latency, memory, and parameter count | measurements on the intended device |

For connected explanations of these choices, open [metrics and tuning](../guides/training/training-quality.md), [vision](../guides/vision/augmentation.md) or [text](../guides/text/tokens-embeddings.md).

### Course 2: what changes in practice

The same loop now supports more deliberate choices. Use these direct routes instead of re-reading a whole course:

| Need to remember | Read |
| --- | --- |
| Precision, recall, macro F1, and metric state | [Metrics](../guides/training/training-quality.md#metrics-that-answer-the-right-question) |
| Step, cosine and plateau LR schedules | [Schedulers](../guides/training/training-quality.md#learning-rate-schedulers) |
| Search spaces, trials and TPE | [Optuna](../guides/training/training-quality.md#a-small-optuna-search) |
| Salt and pepper noise or transform order | [Augmentation](../guides/vision/augmentation.md#noise-as-a-controlled-augmentation) |
| Weights metadata, boxes and masks | [Pretrained vision](../guides/vision/pretrained-models.md) |
| Frozen vs partially fine-tuned features | [Vision stages](../guides/vision/pretrained-models.md#train-the-head-then-fine-tune) |
| Tokenization, padding and vectors | [Text representation](../guides/text/tokens-embeddings.md) |
| EmbeddingBag, class weights and DistilBERT | [Text classifiers](../guides/text/text-classifiers.md) |
| Workers, prefetching and profiling | [Efficient training](../guides/training/efficient-training.md) |
| Mixed precision and accumulation | [Precision](../guides/training/efficient-training.md#mixed-precision) · [Accumulation](../guides/training/efficient-training.md#gradient-accumulation) |

**Metrics:** accuracy can hide a weak minority class. Read per-class results and keep the final test set outside tuning.

**Learning rate:** the optimizer changes weights; a scheduler changes the optimizer's learning rate over time. For an epoch-based `StepLR`, call `scheduler.step()` after that epoch's training updates:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer)  # your training function
    scheduler.step()  # after five calls, multiply the learning rate by 0.1
```

**Transfer learning:** reuse learned visual features and train a new final classifier first. The replacement layer must have one output per class; the optimizer should receive only parameters you intend to train:

```python
from torchvision.models import resnet18, ResNet18_Weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
for parameter in model.parameters():
    parameter.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
preprocess = weights.transforms()  # use for the model's expected input format
```

The frozen backbone still participates in the forward pass; `requires_grad=False` prevents its weights from being updated. Decide on fine-tuning only after checking validation behavior. If you change which layers are trainable, rebuild the optimizer for the new parameter set.

Freezing parameters does not freeze BatchNorm statistics. Use a deliberate [training-mode policy](../guides/vision/pretrained-models.md#frozen-weights-and-evaluation-mode-are-different) for the backbone.

## Function finder

Each row groups related functions by the reason to use them. Follow the link for the shape contract, example and common mistake.

| Function / parameter | Remember | Explanation |
| --- | --- | --- |
| `tensor`, `from_numpy`, `zeros`, `ones`, `arange`, `rand` | creation; copy vs shared storage | [Tensor operations](../guides/fundamentals/core-workflow.md#tensor-operations-worth-remembering) |
| `reshape`, `flatten`, `squeeze`, `unsqueeze` | change dimensions; preserve batch | [Shapes](../guides/fundamentals/core-workflow.md#reshape-without-changing-the-data) |
| `cat`, `stack`, `transpose`, `permute` | join or reorder; not equivalent | [Tensor operations](../guides/fundamentals/core-workflow.md#tensor-operations-worth-remembering) |
| boolean masks, `sum`, `mean`, `std`, `matmul` | selection, reductions and matrix math | [Tensor operations](../guides/fundamentals/core-workflow.md#tensor-operations-worth-remembering) |
| `.to`, `.float`, `.long`, `.item`, `.detach` | device/dtype or Python scalar; gradient history | [Dtype/device](../guides/fundamentals/core-workflow.md#dtype-and-device) |
| `Module`, `Sequential`, `ModuleList` | register layers; define their flow | [Flexible models](../guides/training/training-quality.md#flexible-models-must-register-every-layer) |
| `Linear`, `ReLU`, `Tanh`, `Sigmoid`, `Softmax` | representation and output meaning | [Activations and logits](../guides/fundamentals/core-workflow.md#models-activations-and-logits) |
| `MSELoss`, `L1Loss`, `SmoothL1Loss`, `CrossEntropyLoss`, `BCEWithLogitsLoss`, `NLLLoss` | match task, target shape and output | [Loss table](../guides/fundamentals/core-workflow.md#match-the-output-target-and-loss) |
| `backward`, `zero_grad`, `step`, `SGD`, `Adam`, `AdamW` | calculate then apply gradients | [Update loop](../guides/fundamentals/core-workflow.md#loss-autograd-and-optimizers) |
| `train`, `eval`, `no_grad`, `inference_mode` | module behavior vs gradient tracking | [Evaluation](../guides/fundamentals/core-workflow.md#validation-and-metrics) |
| `Dataset`, `Subset`, `random_split`, `DataLoader` | samples, indices and batches | [Data contracts](../guides/fundamentals/core-workflow.md#dataset-transforms-and-dataloader) |
| `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`, `drop_last` | CPU preparation and batching trade-offs | [Loader settings](../guides/training/efficient-training.md#dataloader-settings) |
| `Conv2d`, `MaxPool2d`, `AdaptiveAvgPool2d` | learned spatial features and reduction | [CNN shapes](../guides/fundamentals/vision-real-data.md#cnn-architecture-and-shapes) |
| `Dropout`, `BatchNorm2d`, `weight_decay` | training-only variation / statistics / regularization | [Hyperparameters](../guides/training/training-quality.md#what-each-hyperparameter-changes) |
| `named_parameters`, `named_children`, `named_modules`, `register_forward_hook` | inspect registered tensors and activations | [Inspection](../guides/fundamentals/vision-real-data.md#inspect-the-model-and-its-activations) |
| `decode_image`, `ToTensor`, `ToPILImage`, `make_grid`, `save_image` | image type, range and visual inspection | [Image utilities](../guides/vision/augmentation.md#pil-tensors-and-image-utilities) |
| `ImageFolder`, `FakeData`, `EMNIST`, `SVHN` | dataset-specific labels and split arguments | [Dataset choices](../guides/vision/augmentation.md#choose-a-dataset-interface) |
| `Compose`, `Resize`, `CenterCrop`, `RandomResizedCrop`, `ColorJitter`, `RandomAffine` | ordered, label-preserving transforms | [Transform pipeline](../guides/vision/augmentation.md#build-the-transform-pipeline-in-order) |
| custom `__call__`, `Normalize` | noise before normalization; fixed mean/std | [Noise](../guides/vision/augmentation.md#an-original-tensor-transform) · [Normalization](../guides/vision/augmentation.md#normalization-and-display) |
| `weights.transforms`, `weights.meta`, `topk`, `requires_grad_` | pretrained input/label contract and adaptation | [Pretrained models](../guides/vision/pretrained-models.md#weights-preprocessing-and-class-names) |
| `draw_bounding_boxes`, `draw_segmentation_masks`, `argmax` | display predictions; choose the correct axis | [Task outputs](../guides/vision/pretrained-models.md#classification-detection-and-segmentation) |
| TorchMetrics `update`, `compute`, `reset`, `average` | aggregate across a measurement interval | [Metrics](../guides/training/training-quality.md#average-over-the-dataset) |
| `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau` | schedules and monitored direction | [Schedulers](../guides/training/training-quality.md#learning-rate-schedulers) |
| Optuna `suggest_*`, `create_study`, `optimize`, `best_params`, `trials_dataframe` | search → validation score → recorded trials | [Optuna](../guides/training/training-quality.md#a-small-optuna-search) |
| `AutoTokenizer`, `BertTokenizerFast`, `DataCollatorWithPadding` | matching IDs, special tokens and padded batches | [Tokenizer](../guides/text/tokens-embeddings.md#use-the-matching-pretrained-tokenizer) |
| `Embedding`, `EmbeddingBag`, `collate_fn`, offsets | vectors, pooling and variable-length inputs | [Text classifiers](../guides/text/text-classifiers.md#embeddingbag-with-offsets) |
| `cosine_similarity`, `PCA` | similarity and limited 2D visualization | [Embedding interpretation](../guides/text/tokens-embeddings.md#similarity-context-and-visualization) |
| `AutoModel`, `AutoModelForSequenceClassification`, `.logits`, `save_pretrained` | contextual vectors vs class predictions | [Text fine-tuning](../guides/text/text-classifiers.md#fine-tune-a-pretrained-text-model) |
| `LightningModule`, `LightningDataModule`, `Trainer`, `self.log` | separate learning logic and orchestration | [Lightning](../guides/training/efficient-training.md#what-lightning-automates) |
| `EarlyStopping`, `ModelCheckpoint`, `fast_dev_run`, callback hooks | stopping, best weights and diagnostics | [Callbacks](../guides/training/efficient-training.md#callbacks-and-scheduler-configuration) |
| `PyTorchProfiler`, `schedule`, `profile_memory`, `record_shapes` | identify expensive work | [Profiling](../guides/training/efficient-training.md#profile-a-short-representative-run) |
| `autocast`, `GradScaler`, `precision`, `accumulate_grad_batches` | numerical precision and update boundaries | [Efficient training](../guides/training/efficient-training.md#mixed-precision) |
| `synchronize`, `reset_peak_memory_stats`, `max_memory_allocated`, `element_size` | measure completion and tensor bytes | [Measurements](../guides/training/efficient-training.md#measure-latency-and-memory) |
| `state_dict`, `save`, `load`, `load_state_dict` | retain weights plus their interpretation | [Saving](../guides/fundamentals/vision-real-data.md#saving-and-restoring) |

## Cheatsheet

### Inspect data and models

```python
print(x.shape, x.dtype, x.device)
print(x.min().item(), x.max().item())
print(labels.unique())
print(model)
```

### Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs, targets = inputs.to(device), targets.to(device)
```

### Train

```python
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)
loss = loss_fn(logits, targets)
loss.backward()
optimizer.step()
```

### Evaluate

```python
model.eval()
with torch.no_grad():
    logits = model(inputs)
    predictions = logits.argmax(dim=1)
```

### Save and restore

```python
torch.save(model.state_dict(), "model.pth")
state = torch.load("model.pth", map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
```

### Count trainable parameters

```python
trainable = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
```

### Fix common random seeds

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

## Common errors

### `mat1 and mat2 shapes cannot be multiplied`

The final input dimension does not match `nn.Linear(in_features, ...)`.

```python
print(x.shape)  # immediately before the linear layer
```

Preserve the batch when flattening: `torch.flatten(x, start_dim=1)`.

### `Expected all tensors to be on the same device`

Move the model, inputs, targets, and newly created helper tensors to the same device. Inspect `.device` at the failing operation.

### `Target N is out of bounds`

For `K` output classes, labels for `CrossEntropyLoss` must normally be integer ids in `0..K-1`. Check one-based source labels and the class mapping.

### Loss does not improve

1. Inspect input values, labels, shapes, and dtypes.
2. Verify the output/loss pairing.
3. Try to overfit one small batch.
4. Confirm parameters receive gradients.
5. Confirm the optimizer owns those parameters.
6. Inspect the learning rate.

### Validation changes unexpectedly

- Call `model.eval()` and disable gradients.
- Remove random validation transforms.
- Use a fixed validation split.
- Keep class mapping and deterministic preprocessing consistent.
- Check for overlap or entity leakage between splits.

### CUDA out of memory

- Reduce batch size first.
- Do not retain computation graphs in Python lists.
- Store `loss.item()` rather than the loss tensor.
- Evaluate inside `torch.no_grad()`.
- Reduce input resolution or model size after measuring the bottleneck.

### NaN loss

- Inspect inputs for NaN or infinity.
- Reduce the learning rate.
- Check logarithms, divisions, and normalization.
- Confirm target dtypes and ranges.
- Clip gradients only when they genuinely explode.

## Debug in this order

1. Print shapes, dtypes, devices, and value ranges.
2. Check labels and class range.
3. Verify model output shape and loss pairing.
4. Confirm train/eval mode.
5. Overfit one tiny batch.
6. Inspect learning curves and class-specific errors.
7. Only then change architecture or regularization.

## Glossary

**Activation**
: The output produced by a layer or nonlinear function.

**Autograd**
: PyTorch's automatic differentiation system.

**Batch**
: A group of samples processed together before one optimizer update.

**Epoch**
: One pass through the training dataset.

**Feature map**
: One channel of activations produced by a convolutional filter.

**Gradient**
: The derivative of the objective with respect to a parameter.

**Logit**
: A raw model score before conversion to a probability.

**Loss**
: A differentiable scalar objective measuring prediction error.

**Parameter**
: A trainable tensor such as a weight or bias.

**Regularization**
: A technique intended to improve generalization rather than only training fit.

**Tensor**
: A multidimensional array carrying shape, dtype, and device information.
