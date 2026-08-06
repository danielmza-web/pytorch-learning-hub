---
title: Build a complete classifier
tags:
  - start
  - training
  - data
last_reviewed: 2026-08-06
---

# 2. Build a complete classifier

This is the smallest end-to-end pattern worth understanding. It turns image files and labels into batches, learns from those batches, then checks performance on held-out examples.

## Data changes shape before the model sees it

![Conceptual data path: an image file is transformed into a channel-first tensor, returned by Dataset, then grouped by DataLoader into a batch](../assets/images/data-pipeline-visual.png)

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
images, labels = next(iter(train_loader))

print(images.shape)  # [32, 3, height, width]
print(labels.shape)  # [32]
```

`Dataset` owns one example and its transforms. `DataLoader` owns batches and training order. Random augmentation belongs in training transforms; validation transforms should stay stable.

## Keep the update order visible

```python
model.train()

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()             # 1. clear old gradients
    logits = model(images)            # 2. predict
    loss = loss_fn(logits, labels)    # 3. measure error
    loss.backward()                   # 4. calculate gradients
    optimizer.step()                  # 5. update weights
```

For single-label image classification, pair raw logits with `nn.CrossEntropyLoss()`. Do **not** apply softmax before that loss. Use `logits.argmax(dim=1)` when you need a predicted class.

## Evaluate is a separate mode

```python
model.eval()
with torch.no_grad():
    logits = model(images)
    predictions = logits.argmax(dim=1)
```

`model.eval()` changes dropout and batch-normalization behaviour. `torch.no_grad()` stops gradient tracking. Use both; neither calculates accuracy by itself.

Next: run the [EMNIST letter classifier](../projects/emnist.md), or continue to [CNNs for images](cnn.md).
