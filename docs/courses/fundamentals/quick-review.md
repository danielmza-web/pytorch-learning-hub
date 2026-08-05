---
title: Fundamentals quick review
course: PyTorch Fundamentals
tags:
  - quick-review
last_reviewed: 2026-08-05
---

# Fundamentals quick review

## 1. Shapes are part of the program

```python
images.shape  # [batch, channels, height, width]
labels.shape  # [batch]
```

For a batch of 32 RGB images at 224 × 224 pixels:

```text
[32, 3, 224, 224]
```

Use the [tensor-shape explorer](../../concepts/tensor-shapes.md) when dimensions are unclear.

## 2. Model and data share one device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
images = images.to(device)
labels = labels.to(device)
```

## 3. Dataset returns one sample; DataLoader returns a batch

```python
class ImageDataset(Dataset):
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = load_image(self.paths[index])
        return self.transform(image), self.labels[index]
```

## 4. Models return logits

```python
logits = model(images)
predictions = logits.argmax(dim=1)
```

Do not apply softmax before `CrossEntropyLoss`.

## 5. Memorize the update order

```python
optimizer.zero_grad()
logits = model(images)
loss = loss_fn(logits, labels)
loss.backward()
optimizer.step()
```

```text
clear → predict → measure → differentiate → update
```

## 6. Evaluation needs two switches

```python
model.eval()
with torch.no_grad():
    logits = model(images)
```

`model.eval()` changes dropout and batch normalization behavior. `torch.no_grad()` disables gradient tracking. Neither replaces the other.

## 7. CNN shape pattern

```text
channels:     3 → 32 → 64 → 128
spatial size: 32 → 16 →  8 →   4
```

Convolution usually increases learned feature channels. Pooling or stride usually reduces spatial resolution.

## 8. Generalization matters more than training accuracy

```text
training loss ↓, validation loss ↓  → learning
training loss ↓, validation loss ↑  → overfitting
both remain poor                  → underfitting or pipeline issue
```

## 9. Save parameters, not only the whole object

```python
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```

## 10. Debug in this order

1. Print shapes and dtypes.
2. Check label range.
3. Check device placement.
4. Verify train/eval mode.
5. Try to overfit one tiny batch.
6. Inspect learning curves.

[Open the complete cheatsheet](../../reference/cheatsheet.md){ .md-button .md-button--primary }

