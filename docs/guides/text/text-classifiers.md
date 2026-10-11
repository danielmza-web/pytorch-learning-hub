---
title: 4.2 · Classifiers and fine-tuning
tags: [text, embeddingbag, distilbert, fine-tuning]
guide_id: "4.2"
last_reviewed: 2026-10-11
---

# 4.2 · Text — classifiers and fine-tuning {#text-classifiers-and-fine-tuning}

Compare a cheap pooled-embedding baseline with a pretrained contextual model. Both still use batches, logits, cross-entropy, gradients and validation.
{ .page-lead }

<div class="learning-context" data-guide-id="4.2" markdown="1">

**Guide 4.2 · Work with text**
{ .learning-location }

**You will learn:** Compare pooled word vectors with a contextual model and retain the complete text input contract.

**Before you start:** Read [4.1 · Tokens and embeddings](tokens-embeddings.md); [3.2 · Pretrained models](../vision/pretrained-models.md) explains the same freeze/adapt principle.

**Practice:** [P7 · Variable-length text classifier](../../projects/text-bags.md)

</div>

**Notation:** `[N,L,E]` means batch, token positions and embedding width; pooled `[N,E]` has one vector per input; logits `[N,K]` have `K` class scores. An offset is a start index into concatenated token IDs. Macro F1 averages the per-class precision/recall score equally, so rare classes remain visible.

**Code key:** “Runnable toy” includes imports and inputs. Other snippets are excerpts: reuse `torch`, `nn`, and the model, loader, tokenizer or helper named in the section. Projects contain the complete runnable scripts.

## Two paths to class logits

```text
word IDs → EmbeddingBag / masked pooling → linear head → [N,K]
subword IDs + attention mask → pretrained transformer → head → [N,K]
```

The first is fast and useful as a baseline, but pooling discards token order. The second can use context and order, with a higher resource cost. Classification assigns a label; it is not text generation, named-entity tagging or a conversation model.

### Follow one batch through both models

For two texts, four token positions after padding, embedding width eight and three classes:

| Stage | Pooled baseline | Contextual classifier |
| --- | --- | --- |
| Token input | IDs `[2,4]` plus padding mask | matching checkpoint IDs/mask `[2,4]` |
| Representation | lookup vectors `[2,4,8]` | contextual hidden states `[2,4,E]` |
| One vector per text | masked pool `[2,8]` | architecture-specific summary `[2,E]` |
| Class output | linear head `[2,3]` | classification head `[2,3]` |
| Training target | class IDs `[2]` | class IDs `[2]` |

Dimensions are illustrative; `E` comes from the chosen checkpoint. The contextual model combines token information before summarizing the text. `AutoModel` normally returns representations, while `AutoModelForSequenceClassification` adds the task head. Neither turns a class ID into a sentence. Return to [transformers and sequence models](../architectures/transformers.md) when the encoder's internal computation is the missing link. [Hugging Face classification guide](https://huggingface.co/docs/transformers/en/tasks/sequence_classification).

## EmbeddingBag with offsets

<div class="offset-strip" aria-label="Two token sequences and their start offsets"><span>offset 0 → <b>2</b> <b>4</b> <b>6</b></span><span>offset 3 → <b>3</b> <b>5</b></span></div>
<p class="visual-caption">Illustration matching the code below: five concatenated IDs, two starts, two pooled vectors. Reversing tokens within either group leaves mean pooling unchanged.</p>

`nn.EmbeddingBag` looks up and pools variable-length groups without creating the full padded `[N,L,E]` intermediate. With 1D input, offsets indicate where each sequence starts:

```python
import torch
from torch import nn

ids = torch.tensor([2, 4, 6, 3, 5], dtype=torch.long)
offsets = torch.tensor([0, 3], dtype=torch.long)
bag = nn.EmbeddingBag(10, 8, mode="mean")
pooled = bag(ids, offsets)  # [2,8]; first bag ids[0:3], second ids[3:5]
logits = nn.Linear(8, 3)(pooled)  # [2,3]
```

This uses the default `include_last_offset=False`: supply one start offset per sample. The last bag continues to the end of `ids`. Do not mix that convention with formats that include the terminal offset.

An original CPU collator, reserving ID 1 for an empty/unknown input:

```python
--8<-- "examples/recall_patterns.py:bag-collate"
```

Use `import torch`. Pass the function as `DataLoader(..., collate_fn=collate_bags)`, then move IDs, offsets and labels to the device in your loop. **Labels describe sequences**, so their count equals the number of offsets, not the total number of token IDs.

| Pooling mode | Mechanism | What to consider |
| --- | --- | --- |
| mean | average token vectors | balances sequence length; order is lost |
| sum | add vectors | longer texts can have larger magnitude |
| max | strongest value per embedding dimension | dimensions can come from different tokens |

The embedding dimension, dropout and head width are tunable settings; no pooling mode wins universally. Compare the same split and training budget.

## Manual pooling must ignore padding

<!-- learning-overview:pooling -->
<section class="learning-overview" data-overview="pooling" aria-label="One set of vectors, three different pools"><p><strong>One set of vectors, three different pools</strong></p><p><code>clear = [2,−2] · image = [4,−4] · PAD = [0,0]</code></p><div class="overview-grid"><div class="overview-card"><b>mean</b><p><strong>[3, -3]</strong></p><p>Average the two real rows.</p></div><div class="overview-card"><b>sum</b><p><strong>[6, -6]</strong></p><p>Add the two real rows.</p></div><div class="overview-card"><b>max</b><p><strong>[4, -2]</strong></p><p>Take the largest real value in each coordinate.</p></div></div><div class="overview-table-wrap"><table class="learning-table"><caption>Why exclude PAD?</caption><thead><tr><th scope="col">Rule</th><th scope="col">PAD excluded</th><th scope="col">PAD included</th></tr></thead><tbody><tr><th scope="row">mean</th><td>[3, -3]</td><td>[2, -2]</td></tr><tr><th scope="row">max</th><td>[4, -2]</td><td>[4, 0]</td></tr></tbody></table></div><p>Reversing “clear image” to “image clear” leaves all three pools unchanged: order is lost. All PAD explicitly returns [0,0] in this toy; a real application may reject empty text or use UNK.</p><p><small>Exact fixed-vector arithmetic; not trained text representations.</small></p></section>
<!-- /learning-overview -->


For padded IDs `[N,L]`, `nn.Embedding` gives `[N,L,E]`. A correct masked mean:

```python
--8<-- "examples/recall_patterns.py:masked-mean"
```

Call it with `masked_mean(embedding(ids), ids)`. Padding changes neither the sum nor the denominator. The helper returns zeros for an all-padding input; alternatively reject empty texts or map them to `<unk>` before training.

For **max** pooling, masked positions must be excluded, commonly with negative infinity. Zeroing them is incorrect when all valid values are negative: padding would win. Handle an all-padding sequence before returning a max, since its result could otherwise be negative infinity. Sum pooling ignores padded positions but remains length-sensitive.

Averaging embeddings means `"camera missed defect"` and a rearrangement of the same tokens have the same pooled representation. If sequence order changes the label, that baseline has a structural limit.

## Class imbalance and validation

Split raw examples before fitting a vocabulary. Stratification can preserve class proportions; group related or duplicate texts together to prevent leakage. Build weights from **training labels**, never validation labels.

A common balanced-weight rule for `K` classes and `N` samples is `N / (K × class_count)`. For known classes `0..K-1`:

```python
counts = torch.bincount(train_labels, minlength=classes).float()
if (counts == 0).any():
    raise ValueError("Every class needs training examples")
weights = counts.sum() / (classes * counts)
loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
```

Here `train_labels` is a CPU long tensor. Weight order must match class IDs. `compute_class_weight("balanced", classes=..., y=...)` is the corresponding scikit-learn utility. Weighting emphasizes errors in rare classes; it does not create missing examples or guarantee better calibration. Avoid combining oversampling and weighting blindly.

Use validation macro F1 and per-class precision/recall alongside accuracy. A rare class can fail while weighted averages remain high. Evaluate new phrasing and the environment where inputs will originate.

When reporting **weighted cross-entropy** for class-index targets, PyTorch's mean reduction divides by the sum of selected target weights, excluding ignored labels. Average epoch loss with that denominator, rather than giving short batches equal weight or multiplying each weighted mean by batch size. Use the same weights and reduction for comparisons. Accuracy/F1 still count the actual predictions and labels; a weighted loss does not automatically make those metrics weighted. [CrossEntropyLoss reductions](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

## Fine-tune a pretrained text model

Hugging Face provides a matching tokenizer and classifier. The example below may download weights and requires Transformers:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=classes,
).to(device)
```

Loading a base checkpoint adds a new classification head. A warning about newly initialized head weights is expected: the head needs training. It is not evidence that the model already knows your target labels.

Each Dataset item tokenizes one text with truncation and returns the fields plus a label. `DataCollatorWithPadding` batches them. Given such a loader and an optimizer owning the intended trainable parameters:

```python
model.train()
for batch in train_loader:
    batch = {key: value.to(device) for key, value in batch.items()}
    labels = batch.pop("labels")
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)               # unpack input_ids and attention_mask
    loss = loss_fn(outputs.logits, labels) # your class-weighted loss, if selected
    loss.backward()
    optimizer.step()
```

`.logits` has shape `[N,K]`. Alternatively, passing `labels` into the model can produce its built-in `.loss`; do not assume that built-in loss applies your external class weights. Use one consistent loss path. DistilBERT does not accept the same fields as every other transformer; use its matching tokenizer rather than inventing `token_type_ids`.

## Partial fine-tuning with DistilBERT

The same [vision transfer strategies](../vision/pretrained-models.md#three-transfer-learning-strategies) apply. Freeze everything, then explicitly select a few final transformer blocks plus the classification layers:

??? note "Implementation excerpt · requires the objects described above"
    ```python
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layers = model.distilbert.transformer.layer
    last_blocks = 2
    if not 0 <= last_blocks <= len(layers):
        raise ValueError("Invalid number of blocks")
    selected = list(layers)[len(layers) - last_blocks:]
    for module in selected + [model.pre_classifier, model.classifier]:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=2e-5,
    )
    ```


This is specifically for a DistilBERT sequence classifier. Other architectures have different module paths. `last_blocks=0` trains only the head; avoid `layers[-0:]`, which would select every layer. Count and inspect trainable parameters before starting.

Partial fine-tuning can reduce backward and optimizer cost. The frozen blocks still run in the forward pass. `requires_grad=False` does not disable Dropout: decide whether frozen blocks stay in training or evaluation mode and apply that policy each epoch. Full fine-tuning can improve adaptation but also raises memory requirements, overfitting and catastrophic-forgetting risk. Compare rather than assuming it wins.

## Evaluate and retain the whole input contract

```python
model.eval()
batch = tokenizer(["Check the surface again."], truncation=True,
                  max_length=128, return_tensors="pt")
batch = {key: value.to(device) for key, value in batch.items()}
with torch.inference_mode():
    predicted_id = model(**batch).logits.argmax(dim=1).item()
```

Interpret the ID with your saved class map. Save model and tokenizer with `save_pretrained(...)`, plus label mapping, text policy, sequence limit, split/seed, versions and selected validation result. A simple pooled classifier needs its vocabulary and architecture saved alongside `state_dict()`.

**Check yourself:** only the classifier was unfrozen, but the optimizer was created before replacing that head. Will the new head learn? It may not: confirm that the optimizer contains the new parameters, then rebuild it.

**Run the baseline:** [Variable-length text classifier](../../projects/text-bags.md) shows a training-only vocabulary, offsets, mean pooling, class weights and saved predictions in one original script with no downloads.

For the next experiment, compare one baseline, one partial fine-tune and one full fine-tune using quality, time and memory. [Efficient pipelines](../training/efficient-training.md) explains the measurement and accumulation choices.

After selecting the model, use [model lifecycle](../deployment/model-lifecycle.md) to package its input contract, then [optimization and serving](../deployment/optimization-serving.md) to measure CPU latency and check that a smaller artifact preserves acceptable predictions.

**Sources:** [EmbeddingBag](https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html), [DistilBERT](https://huggingface.co/docs/transformers/en/model_doc/distilbert), [Hugging Face text classification](https://huggingface.co/docs/transformers/en/tasks/sequence_classification).

**Continue reading:** [5.1 · Vision, similarity and generation](../architectures/vision-generative.md). Return to [the learning map](../../index.md#choose-a-guide) to choose another topic.
