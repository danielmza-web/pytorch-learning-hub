---
title: Text — tokens, embeddings and batches
tags: [text, tokenizer, embeddings, padding]
last_reviewed: 2026-10-01
---

# Text — tokens, embeddings and batches

Text has no built-in numeric representation. Tokenization supplies discrete IDs; embeddings supply vectors; batching supplies a consistent input shape. These are separate operations.
{ .page-lead }

```text
raw text → tokens → integer IDs [N,L] → embeddings [N,L,E]
                  + mask                    ↓
                                    pooling or contextual model
```

Start here before [text classifiers](text-classifiers.md).

**Code key:** “Runnable toy” includes imports and inputs. Other snippets are excerpts: reuse `torch`, `nn`, and the model, loader, tokenizer or helper named in the section. Projects contain the complete runnable scripts.

## Tokens are not meaning

<div class="recall-flow" role="group" aria-label="Input to output">
<div><b>Words</b><code>clear image / bad blurry photo</code><small>lengths 2 and 3</small></div>
<div><b>Toy vocabulary</b><code>clear=2, image=3, bad=4, blurry=5, photo=6</code><small>IDs are keys, not numeric meaning</small></div>
<div><b>Padded batch</b><code>[2,3,0] / [4,5,6]</code><small>mask [1,1,0] / [1,1,1]</small></div>
<div><b>Embedding lookup</b><code>[2,3,E]</code><small>one vector per token position</small></div>
<div><b>Masked mean</b><code>[2,E]</code><small>divide each sum by its real token count</small></div>
</div>
<p class="visual-caption">Toy vocabulary illustration. The runnable project has its own saved training-only vocabulary; pretrained tokenizer IDs depend on the checkpoint.</p>

| Tokenization | What gets an ID | Trade-off |
| --- | --- | --- |
| Word | a word such as `inspection` | interpretable, but a large vocabulary and unknown words |
| Subword | reusable pieces of words | handles many rare words; may expand sequence length |
| Character | individual characters | small vocabulary; longer sequences |

A token ID is a lookup key: ID 200 is not “twice as meaningful” as ID 100. Do not feed IDs to a linear layer as ordinary numeric features.

For a tiny word-based baseline, an original vocabulary example:

```python
training_texts = ["camera finds scratches", "camera checks panels"]
vocab = {"<pad>": 0, "<unk>": 1}
for text in training_texts:
    for token in text.lower().split():
        if token not in vocab:
            vocab[token] = len(vocab)

def encode(text):
    ids = [vocab.get(token, 1) for token in text.lower().split()]
    return ids or [1]  # explicitly handle an empty/cleaned-away sentence
```

Build your vocabulary from training data only. Reuse the same mapping for validation and new inputs. A minimum frequency removes rare vocabulary entries at the cost of more `<unk>` tokens. Save the tokenizer/cleaning rules and vocabulary with the model.

Lowercasing, punctuation removal and whitespace splitting are baseline choices. They can erase useful distinctions: `not`, numbers, accented letters, product codes and punctuation can carry meaning. Avoid custom cleaning that conflicts with a pretrained tokenizer.

### Special tokens

`<pad>` fills unused positions; `<unk>` represents unrecognized input. Beginning/end tokens mark boundaries when required. BERT-style `[CLS]` and `[SEP]` are part of that model's input convention; the tokenizer normally inserts them. Subword vocabularies reduce the unknown-word problem but do not guarantee that every input is understood.

## Use the matching pretrained tokenizer

The tokenizer and model must share their vocabulary, normalization and special-token convention. BERT and DistilBERT are related, but use the exact tokenizer belonging to the chosen checkpoint.

```python
from transformers import AutoTokenizer

checkpoint = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
batch = tokenizer(
    ["Inspect the panel.", "The camera missed a small mark."],
    padding=True, truncation=True, max_length=128, return_tensors="pt",
)
print(batch["input_ids"].shape)       # [N,L], integer IDs
print(batch["attention_mask"])        # 1 = real token; 0 = padding
print(tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist()))
```

This may download tokenizer files. `AutoTokenizer` chooses the implementation for the checkpoint; `BertTokenizerFast` explicitly chooses a BERT fast tokenizer. `.tokenize(text)` shows pieces, `.convert_ids_to_tokens(...)` helps inspect encoded IDs, and `.get_vocab()` exposes the mapping. The precise pieces and IDs are checkpoint-dependent; do not memorize sample IDs.

## Padding, truncation and attention masks

Default collation cannot stack vectors with different lengths. Choose a representation suitable for the model:

| Choice | What changes |
| --- | --- |
| Fixed padding | every example is padded to a chosen maximum |
| Dynamic padding | each batch pads only to its longest sequence |
| Length bucketing | similar-length sequences share batches, reducing waste |
| Truncation | cuts inputs exceeding `max_length`; useful text can be lost |
| Flattened IDs + offsets | `EmbeddingBag` pools variable-length inputs without padding |

For lengths 3, 8 and 16, dynamic padding uses 48 positions, of which 21 are padding. Padding to 64 uses 192 positions. That arithmetic is not a speed benchmark: transformer attention and other operations have additional costs, and an attention mask does not remove all padded computation.

<div class="interactive-panel" data-padding-lab>
  <div class="interactive-heading">Count padding before training</div>
  <label>Sequence lengths <input value="3,8,16" data-sequence-lengths aria-label="Comma separated positive token counts"></label>
  <label>Fixed maximum <input type="number" min="1" max="4096" value="64" data-fixed-length></label>
  <output data-padding-output aria-live="polite">Dynamic: 48 positions, 21 padding. Fixed length 64: 192 positions, 165 padding.</output>
  <small>Synthetic token counts, including any special tokens. Shows truncation when the maximum is too small.</small>
</div>

Use `padding=True` for the longest sequence in the current tokenizer call, or `padding="max_length"` for fixed padding. `truncation=True` and an explicit `max_length` bound the input; choose the bound with the model's limit and the text length distribution in mind. A DistilBERT base checkpoint typically supports up to 512 positions, including special tokens.

An **attention mask** identifies valid positions, usually with 1, and padding with 0. It is not a class label. For manual embedding pooling, you must exclude padding yourself. For transformer batches, pass it to the model.

### Pad at collation time

```python
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
loader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True,
                    collate_fn=collator)
```

Each `tokenized_dataset` sample should contain unpadded tokenization fields and an integer `labels` value. The collator builds `[N,L]` tensors at batch time. Keep collation on CPU; move the finished batch to the model device in the training loop. This also supports DataLoader workers.

## Embeddings are learned lookups

`nn.Embedding(V,E)` stores a trainable table with `V` rows and `E` values per row. It maps IDs to vectors; it does not create contextual understanding by itself.

```python
import torch
from torch import nn

table = nn.Embedding(12, 5, padding_idx=0)
ids = torch.tensor([[2, 3, 0], [4, 1, 5]], dtype=torch.long)
vectors = table(ids)
print(vectors.shape)  # [2,3,5]: batch, length, embedding dimension
```

The vectors start from initialization unless pretrained values are loaded. Gradients from the downstream objective teach useful entries. `padding_idx=0` suppresses normal gradient updates for the padding row; masking still matters when pooling or after custom initialization.

### Alternative representations

Counts, TF-IDF, learned word vectors and contextual encoders retain different information about text.

??? note "Code and details"
    | Representation | What it captures | What it loses / costs |
    | --- | --- | --- |
    | One-hot | identity in a vocabulary-sized vector | sparse; no learned similarity |
    | Bag of words | token occurrence/count | word order |
    | TF-IDF | counts weighted down for common terms | context and order; fit on training data |
    | Static vectors: GloVe, Word2Vec, FastText | learned distributional relationships | a fixed vector per token, with model-specific OOV behavior |
    | Contextual model: BERT/DistilBERT | representations influenced by surrounding tokens | more computation and memory |

    A small learned embedding can be sufficient for a short-label task. A pretrained language model can help with context, but validation must establish that benefit for your data.

## Similarity, context and visualization

Cosine similarity compares direction, not vector length:

```python
import torch.nn.functional as F
similarity = F.cosine_similarity(vector_a, vector_b, dim=-1)
```

Near 1 means aligned vectors; near 0 means nearly orthogonal; near -1 means opposed directions. It does not universally mean synonyms, unrelated concepts and antonyms respectively. Antonyms can occur in similar contexts and have similar embeddings.

GloVe returns the same stored vector for a word regardless of sentence. A contextual model can distinguish `bat` as an animal from sports equipment. `AutoModel` exposes `.last_hidden_state` with shape `[N,L,E]`; select the actual token position or aggregate its subwords. Do not assume a word always occupies a fixed index, and do not include padding or special tokens by accident.

The studied toy embedding exercise predicts context words from a centre word. Repeated co-occurrence and cross-entropy updates shape the lookup table. A tiny toy corpus demonstrates the mechanism; it does not establish broad semantic knowledge. Vector analogies are illustrative relationships, not guaranteed facts.

PCA (`PCA(n_components=2).fit_transform(vectors)`) projects onto two directions of high variance; t-SNE emphasizes local neighborhoods. Both distort the original space. A nearby pair in a two-dimensional plot is not sufficient evidence of semantic accuracy. Check similarity in the full vector space and evaluate the downstream task.

## Remember and apply

**Check yourself:** does zero-padding reduce an unmasked mean embedding? Yes: dividing by total padded length changes the vector's magnitude. The mask must determine both the sum and its denominator.

Continue with [pooled and pretrained text classifiers](text-classifiers.md), where the same IDs and masks connect to logits, loss and fine-tuning.

**Sources:** [Hugging Face tokenizers](https://huggingface.co/docs/transformers/en/main_classes/tokenizer), [padding collator](https://huggingface.co/docs/transformers/en/main_classes/data_collator), [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html), [GloVe](https://nlp.stanford.edu/projects/glove/).
