---
title: Variable-length text classifier
study_context: Course 2 text workflows
tags: [project, text, embeddings]
last_reviewed: 2026-10-01
---

# Project: variable-length text classifier

Train a small classifier from original image-description phrases. Follow vocabulary → token IDs → offsets → mean embeddings → class scores.
{ .page-lead }

## What to remember

Different-length sentences do not require a fixed padded matrix. `EmbeddingBag` accepts concatenated IDs plus the start offset of each sentence and returns one pooled vector per sentence.

This is an inexpensive baseline, not a contextual language model. It combines the course's tokenization, vocabulary, collation, pooling and class-weight ideas without external course helpers or pretrained downloads.

## Important code

```python
ids = torch.tensor([2, 3, 4, 5, 6], dtype=torch.long)
offsets = torch.tensor([0, 2], dtype=torch.long)
embedding = nn.EmbeddingBag(vocab_size, 12, mode="mean", padding_idx=0)
head = nn.Linear(12, 2)
logits = head(embedding(ids, offsets))  # [2, 2]
```

The first sentence is `[2,3]`, the second is `[4,5,6]`. Their lengths differ, but both produce a 12-value vector. Offsets use start positions; the default API does not require a final ending offset.

<div class="recall-flow" role="group" aria-label="Input to output">
<div><b>Two sentences</b><code>lengths 2 and 3</code><small>five token IDs</small></div>
<div><b>Concatenate</b><code>offsets [0,2]</code><small>two starts, no padding</small></div>
<div><b>EmbeddingBag</b><code>[2,12]</code><small>one mean vector per sentence</small></div>
<div><b>Head</b><code>[2,2]</code><small>two class scores</small></div>
<div><b>Loss</b><code>scalar</code><small>class-weighted cross-entropy</small></div>
</div>
<p class="visual-caption">Illustration: shapes and operations, not measured model performance.</p>


| Choice | This example | Why it matters |
| --- | --- | --- |
| Vocabulary | Sorted words from training text only | Prevents validation text from defining the representation. |
| Reserved IDs | Padding `0`, unknown `1` | Unknown words and padding mean different things. |
| Empty input | One unknown token | Makes the empty-input policy explicit. |
| Collation | Flat IDs, start offsets and integer labels | Keeps variable-length batching on CPU. |
| Class weights | Inverse training counts, aligned with IDs | Gives the rarer class more weight in the loss. |
| Pooling | Mean | Balances length but discards word order. |

The tiny training set has six positive and two negative descriptions. It computes weights `[2.0, 2/3]` in label order. More weight changes the loss priorities; it does not guarantee better minority-class predictions.

## Check the representation

If you reverse the words in a sentence, its mean-pooled vector stays the same. “Dog bites person” and “person bites dog” therefore cannot be distinguished by this baseline.

A padded `nn.Embedding` alternative needs a mask when computing a manual mean. Compare its padding cost below; the script itself uses offsets and no padding:

<div class="interactive-panel" data-padding-lab>
  <div class="interactive-heading">Offsets avoid these padding positions</div>
  <label>Sequence lengths <input value="2,3" data-sequence-lengths aria-label="Comma separated positive token counts"></label>
  <label>Fixed maximum <input type="number" min="1" max="4096" value="8" data-fixed-length></label>
  <output data-padding-output aria-live="polite"></output>
  <small>Synthetic token counts; this compares padded batching, not measured runtime.</small>
</div>

## Run and inspect

```bash
python examples/text_bags.py
python examples/text_bags.py --epochs 40 --output artifacts/my-text-run
```

The offline CPU run saves `predictions.json` and `text-model.pt` under the output directory. Inspect the token map, class weights, validation history and predictions for four new combinations of words. The checkpoint includes the vocabulary, label names and tokenization/empty-input policy.

These toy phrases demonstrate the data flow. Their validation scores are not a sentiment benchmark and do not prove useful performance on natural language. For a real application, enlarge the dataset and inspect per-class recall and macro F1.

**Try changing:** add a new training phrase, rerun vocabulary construction, then compare a new word with an unknown word. If order changes the correct label, move to an order-aware or contextual model rather than tuning a bag of words indefinitely.

[Review tokenization and embeddings](../guides/text/tokens-embeddings.md) · [Review classifiers and fine-tuning](../guides/text/text-classifiers.md)

## Inspect a retained tiny execution

**Measured toy execution, CPU, 20 epochs, seed 53.** Eight training phrases and four validation phrases are too small to establish general sentiment quality. Here the labels describe image quality.

| Word | ID | First four values of its 12-value learned row |
| --- | --- | --- |
| `<pad>` | 0 | `0.000, 0.000, 0.000, 0.000` |
| `<unk>` | 1 | `-0.959, 0.895, -0.575, 0.478` |
| `bad` | 2 | `0.441, -0.121, 0.357, -1.113` |
| `blurry` | 3 | `-0.764, -1.040, 0.530, 1.699` |
| `bright` | 4 | `-0.896, 2.745, -1.294, 0.934` |
| `clear` | 5 | `2.665, 1.200, -0.462, 0.496` |
| `dark` | 6 | `-2.493, 0.230, 1.268, 1.275` |
| `good` | 7 | `0.557, 0.951, 1.377, 0.468` |
| `image` | 8 | `-0.549, 0.536, 0.037, 0.650` |
| `photo` | 9 | `-1.242, 0.326, -0.902, 0.074` |
| `picture` | 10 | `0.071, 0.787, -1.774, -0.047` |
| `sharp` | 11 | `1.112, 0.490, -0.195, 0.725` |

The first two phrases `clear image` and `sharp picture` become flat IDs `[5, 8, 11, 10]` and offsets `[0, 2]`. Each offset starts a segment; the final segment runs to the end. Mean pooling turns each segment into one 12-value row; the classifier returns two logits per phrase. No PAD rows are needed for this bag representation.

Training class counts are `[2, 6]`; inverse-frequency weights are `[2.0, 0.6667]`. For class-index cross-entropy, the mean divides weighted losses by the sum of target weights, not by batch size. [PyTorch loss contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

| Validation phrase | True label | Actual prediction |
| --- | --- | --- |
| good sharp image | 1 | 1 |
| bad dark picture | 0 | 0 |
| clear bright photo | 1 | 1 |
| blurry image | 0 | 0 |

Labels: `0 = poor-image-description`, `1 = good-image-description`. Unknown input `unseenword` becomes `[1]` and predicts `1` in this run. The unknown row has no supervised training examples here; this prediction is not evidence of understanding an unknown word. Empty input also becomes `<unk>`.

`clear image` and `image clear` have maximum logit difference **0.0**: a mean bag loses order even though the original texts differ. Their actual logits are `[-2.6688294410705566, 2.805745840072632]`. Use an order-aware model when the distinction changes meaning. [Retained vocabulary, IDs, offsets, embeddings and predictions](../assets/data/recall-2026-10-01/predictions.json).

## Complete source

??? note "Open the maintained runnable script"
    ```python
    --8<-- "examples/text_bags.py"
    ```

??? note "Open the shared variable-length collator"
    ```python
    --8<-- "examples/recall_patterns.py:bag-collate"
    ```

API reference: [EmbeddingBag inputs, offsets and padding](https://docs.pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html).
