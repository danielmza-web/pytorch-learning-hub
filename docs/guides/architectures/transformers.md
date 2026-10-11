---
title: 5.2 · Attention and transformers
tags: [transformers, attention, encoder, decoder, masking, generation]
guide_id: "5.2"
last_reviewed: 2026-10-11
---

# 5.2 · Transformers — attention, encoders and generation {#transformers-attention-encoders-and-generation}

Tokenization chooses discrete IDs, embeddings turn IDs into vectors, and attention lets those vectors exchange information. Masks and output heads determine whether the result serves classification, next-token generation or sequence transformation.
{ .page-lead }

<div class="learning-context" data-guide-id="5.2" markdown="1">

**Guide 5.2 · Explore advanced architectures**
{ .learning-location }

**You will learn:** Trace positions, attention, masks and output heads through encoder and generation workflows.

**Before you start:** Read [4.1 · Tokens and embeddings](../text/tokens-embeddings.md); [5.1 · Vision, similarity and generation](vision-generative.md) introduces shared modules and residual paths.

**Practice:** Use the visual checks and runnable toys on this page, then [check your understanding](#remember-and-apply). These are architecture demonstrations; the existing projects do not train a diffusion or transformer model.

</div>

**Terms:** scaled dot-product attention (SDPA) mixes value vectors using query–key scores. A feed-forward network (FFN) transforms each position; GELU is the Gaussian error linear unit activation. BOS/EOS mark sequence boundaries, PAD marks padding. “Encoder”, “decoder” and “memory” below describe computation roles, rather than saved files.

Shared modules and residual paths from [5.1 · Advanced vision](vision-generative.md) reappear here. Follow one token position through each calculation; the synthetic examples demonstrate information flow, not language quality.

**Code key:** runnable toys include imports and inputs and use only installed PyTorch. Excerpts depend on the objects explicitly named in their section. No pretrained download or language-quality result is implied.

## Add order to token embeddings

Let `N` be batch size, `L` sequence length, `D` model width, `H` heads and `V` vocabulary size. Integer token IDs have shape `[N,L]`; `nn.Embedding(V,D)` returns `[N,L,D]`. Identical IDs initially select the same vector, wherever they occur. Self-attention without positional information cannot infer ordinary word order from a set of embeddings alone.

A learned position table supplies one `[D]` vector per position, added to the token vector. A fixed sinusoidal scheme instead uses `sin(pos / 10000**(2i/D))` on even dimensions and the corresponding cosine on odd dimensions. Store fixed tables with `register_buffer` so device moves and checkpoints handle them; their lack of gradients is intentional. A model's supported context length must match its positional mechanism and training; sinusoidal values at a new position do not prove reliable length extrapolation. Relative positions and rotary embeddings are other mechanisms used by particular models. [Original transformer paper](https://arxiv.org/abs/1706.03762).

```text
token IDs [N,L] ── embedding table ──┐
                                    + ── input vectors [N,L,D]
positions 0…L-1 ── position signal ─┘
```

Padding is a separate issue. `padding_idx` in an embedding controls normal updates to one table row; it does not make a later attention layer ignore that position. The attention mask and eventual loss/pooling rule must also respect padding.

## Queries, keys and values form attention

Three learned projections of the input produce **queries** `Q`, **keys** `K` and **values** `V_values`. A query is compared against each key; the resulting weights combine value vectors. These names describe roles in a learned calculation, not literal questions or meanings assigned to each vector component.

For one head with key/query width `d_k`:

`attention(Q,K,V_values) = softmax(Q @ K.transpose(-2,-1) / sqrt(d_k) + mask) @ V_values`.

Rows correspond to query positions and columns to key positions. Softmax is taken over keys (`dim=-1`), so the un-dropped weights of each valid row sum to one. Scaling controls score magnitude as width grows. If every key in a query row is blocked, the ordinary softmax of all negative infinities is undefined; design masks to leave a real key available for meaningful queries. [Scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

<div class="recall-flow" role="group" aria-label="Attention tensor shapes">
<div><b>Input</b><code>[N,L,D]</code><small>token vectors with position information</small></div>
<div><b>Project</b><code>Q, K, V_values</code><small>different learned roles</small></div>
<div><b>Compare</b><code>[N,H,L,S]</code><small>one score per query/key pair</small></div>
<div><b>Mask and normalize</b><code>softmax over S keys</code><small>allowed information gets weights</small></div>
<div><b>Combine</b><code>[N,L,D]</code><small>mix values, merge heads, project</small></div>
</div>
<p class="visual-caption">Shape illustration for multihead attention. In self-attention S=L; in cross-attention the source and target lengths can differ.</p>

**Runnable toy — actual attention arithmetic and a causal boundary:**

```python
import math
import torch
import torch.nn.functional as F

# Original fixed vectors: no learned semantics or tokenizer required.
q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
k = q.clone()
values = torch.tensor([[[2.0, 0.0], [0.0, 4.0], [3.0, 3.0]]])
scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
allowed = torch.ones(3, 3, dtype=torch.bool).tril()
weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
output = weights @ values
assert weights.shape == (1, 3, 3) and output.shape == (1, 3, 2)
assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 3))
assert torch.equal(output[0, 0], values[0, 0])  # first query sees only key 0
assert torch.equal(weights[0].triu(diagonal=1), torch.zeros(3, 3))
# SDPA uses True for allowed positions; add the single head dimension.
fused = F.scaled_dot_product_attention(
    q.unsqueeze(1), k.unsqueeze(1), values.unsqueeze(1),
    attn_mask=allowed, dropout_p=0.0,
).squeeze(1)
assert torch.allclose(output, fused, atol=1e-6)
print(weights[0].round(decimals=3))
```

This computes a weighted sum, not a trained language prediction. `scaled_dot_product_attention` expects already projected queries/keys/values; it does not create the projection layers. Its `dropout_p` applies even when surrounding modules are in evaluation mode, so pass `0.0` for deterministic evaluation or choose it according to `self.training`.

## Multihead attention and the surrounding block

`nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True)` includes input/output projections and returns outputs `[N,L,D]`. `D` must be divisible by `H`; each standard head has width `D/H`. With `average_attn_weights=False`, returned weights have shape `[N,H,L,S]`. Heads can learn different relationships, but they are not assigned grammatical jobs and need not become distinct. Set `need_weights=False` when no weight visualization is required. [MultiheadAttention API](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).

Attention mixes positions. A feed-forward network transforms each position independently with shared weights, for example `Linear(D,4D) → GELU → Linear(4D,D)`. Residual additions preserve the width and give direct paths across these transformations. LayerNorm normalizes features within a token representation, then applies learned scale/shift; it is not BatchNorm over a batch. Dropout regularizes the configured branches during training.

In a pre-norm block, a simplified sequence is `x ← x + attention(norm(x))`, then `x ← x + feed_forward(norm(x))`. Post-norm places normalization after addition. PyTorch's `norm_first` selects that placement; model width, head count, feed-forward width, layer count and dropout change capacity and computation. Keep these settings with a saved model's architecture contract.

Standard dense attention compares `L×L` pairs per head in self-attention. Doubling length quadruples this pair count, while embeddings/feed-forward work also grows. Efficient kernels can avoid materializing the whole matrix, and performance depends on hardware; the pair-count argument is not a measured RAM or latency result.

## Three architectures use the same primitives differently

| Architecture | Attention access | Typical head and use |
| --- | --- | --- |
| Encoder | all valid source positions, usually bidirectional | pooled classification or per-token tagging; BERT-style models |
| Decoder-only | the current input position and earlier positions | vocabulary logits for next-token prediction; GPT-style models |
| Encoder-decoder | bidirectional source; causal target; target attends to encoded source | target vocabulary logits for translation/summarization; T5-style models |

These are common designs, not definitions of every model's training objective. An encoder produces sequence representations `[N,S,D]`, not necessarily a fixed vector. A task head can pool those representations or predict per token. Decoder-only models can also support classification through suitable heads; generation does not guarantee factual understanding.

PyTorch's `TransformerEncoderLayer` implements self-attention plus feed-forward operations. With a causal mask and a vocabulary head it can form a small decoder-only language model. `TransformerDecoderLayer` also includes attention to a separate `memory` input, which is appropriate for encoder-decoder models. The API class name alone does not decide the information flow. [PyTorch Transformer interface](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html).

## Masks are contracts, and APIs disagree about booleans

A **padding mask** removes non-content keys. A **causal mask** blocks future target positions. A **loss mask** removes targets that should not contribute to the training objective. These solve different problems and can all be necessary.

| Interface | Meaning of the indicated mask value |
| --- | --- |
| `nn.MultiheadAttention` boolean `attn_mask` | `True` blocks the query/key pair |
| PyTorch transformer boolean padding masks | `True` ignores that key position |
| `F.scaled_dot_product_attention` boolean `attn_mask` | `True` allows the query/key pair |
| Floating additive attention mask | `0` leaves score unchanged; `-inf` blocks it |
| Hugging Face tokenizer `attention_mask` | commonly `1` is a real token and `0` is padding |

For self-attention with `batch_first=True`, a shared causal mask is `[L,L]`; a key padding mask is `[N,L]`. Keep both boolean if using both with PyTorch transformer modules. A padding mask blocks attention **to keys**; it does not guarantee zero output vectors at padded query positions. Exclude those outputs when pooling or scoring a loss.

The causal mask allows the diagonal. At input position 2, the model sees input token 2 and earlier tokens to predict target token 3. This is why shifting the target and masking future input positions work together:

```text
Input:    BOS     inspect   panel    EOS     PAD
Target:   inspect panel     EOS      PAD     PAD

Allowed keys for the first three input positions:
          BOS  inspect  panel
BOS        ✓      ·      ·
inspect    ✓      ✓      ·
panel      ✓      ✓      ✓
```

Here `✓` means allowed, irrespective of the API's boolean convention. Do not assume `is_causal=True` is a drop-in replacement in every API/version: in higher-level transformer APIs it can be a hint about the supplied mask. The examples here supply an explicit mask. Avoid left-padded/all-padding rows in these small generic toys because some query rows can otherwise have no available keys. [Mask semantics in SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [tokenizer masks](https://huggingface.co/docs/transformers/en/main_classes/tokenizer).

<!-- learning-overview:attention -->
<section class="learning-overview" data-overview="attention" aria-label="What can query 1 read?"><p><strong>What can query 1 read?</strong></p><div class="overview-grid"><div class="overview-card"><b>Bidirectional</b><figure class="learning-matrix" tabindex="0" aria-label="Rows = queries · columns = keys. Arrow keys scroll a wide matrix."><figcaption>Rows = queries · columns = keys</figcaption><div class="learning-cells" style="--columns:3"><span>✓</span><span>✓</span><span>✓</span><span class="selected-cell">✓</span><span class="selected-cell">✓</span><span class="selected-cell">✓</span><span>✓</span><span>✓</span><span>✓</span></div></figure><p>Query 1 reads keys 0, 1 and 2.</p></div><div class="overview-card"><b>Causal</b><figure class="learning-matrix" tabindex="0" aria-label="Rows = queries · columns = keys. Arrow keys scroll a wide matrix."><figcaption>Rows = queries · columns = keys</figcaption><div class="learning-cells" style="--columns:3"><span>✓</span><span>×</span><span>×</span><span class="selected-cell">✓</span><span class="selected-cell">✓</span><span>×</span><span>✓</span><span>✓</span><span>✓</span></div></figure><p>Query 1 reads keys 0 and 1; future key 2 is blocked.</p></div></div><figure class="overview-flow"><figcaption>One attention calculation</figcaption><ol><li><b>Scores</b><span>Compare a query with each key.</span></li><li><b>Mask</b><span>Block keys that this query may not read.</span></li><li><b>Softmax</b><span>Turn permitted scores into weights that sum to 1.</span></li><li><b>Mix values</b><span>Use those weights to combine the value vectors.</span></li></ol></figure><p><small>✓ = permitted, × = blocked. Diagram has no PAD. API boolean conventions can differ.</small></p></section>
<!-- /learning-overview -->

<details class="learning-exploration"><summary>Explore the attention calculation</summary>
<div data-learning-lab="attention"><p>Follow fixed query/key scores through masking, softmax and a weighted value sum. Enable JavaScript to adjust this original toy.</p></div>
</details>

## Train next-token prediction with shifted targets

With tokenized examples including boundaries, use `inputs = ids[:, :-1]` and `targets = ids[:, 1:]`. Teacher forcing supplies the real preceding tokens during training; a causal mask lets all target positions be processed in parallel without seeing future input tokens. At inference, the preceding tokens increasingly come from the model's own choices.

The vocabulary head returns raw logits `[N,L,V]`. Flatten to `[N*L,V]` with targets `[N*L]`, or transpose logits to `[N,V,L]` for cross-entropy. Exclude PAD targets with `ignore_index`; do not softmax before `cross_entropy`. An entirely ignored target batch needs an explicit skip/handling rule rather than assuming the mean loss is finite.

**Runnable toy — one update and a real check that future tokens cannot affect earlier outputs:**

```python
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(11)
PAD, BOS, EOS = 0, 1, 2

class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.words = nn.Embedding(8, 8, padding_idx=PAD)
        self.positions = nn.Embedding(8, 8)
        self.block = nn.TransformerEncoderLayer(
            d_model=8, nhead=2, dim_feedforward=16,
            dropout=0.0, batch_first=True,
        )
        self.head = nn.Linear(8, 8)

    def forward(self, ids):
        length = ids.size(1)
        if length > 8:
            raise ValueError("This toy supports at most 8 input positions")
        pos = torch.arange(length, device=ids.device)
        x = self.words(ids) + self.positions(pos).unsqueeze(0)
        blocked = torch.ones(length, length, device=ids.device,
                             dtype=torch.bool).triu(diagonal=1)
        x = self.block(x, src_mask=blocked, src_key_padding_mask=ids.eq(PAD))
        return self.head(x)

tokens = torch.tensor([[BOS, 3, 4, 6, EOS], [BOS, 5, EOS, PAD, PAD]])
inputs, targets = tokens[:, :-1], tokens[:, 1:]
model = TinyCausalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
logits = model(inputs)
assert logits.shape == (2, 4, 8)
loss = F.cross_entropy(logits.reshape(-1, 8), targets.reshape(-1),
                       ignore_index=PAD)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

model.eval()
changed = inputs.clone()
changed[0, -1] = 7  # perturb a future position for earlier queries
with torch.inference_mode():
    before, after = model(inputs), model(changed)
assert torch.allclose(before[0, :3], after[0, :3], atol=1e-6)
assert torch.isfinite(loss)
print("finite loss; earlier logits unaffected by a future token")
```

This checks masking and a finite gradient update; one synthetic step teaches no useful language. Build vocabularies on training text, split by document/source before creating overlapping windows, and avoid nearly identical windows crossing train/validation/test. Select checkpoints with validation loss and evaluate the held-out test once. Perplexity is `exp(mean token negative log-likelihood)` over the scored tokens; comparisons require compatible tokenization, evaluation data and masking. It is not a truthfulness score.

## Generate by appending one token at a time

At generation time, get logits for the last valid input position, choose a next token, append it and repeat until EOS or a length limit. For the untrained toy above, this original excerpt merely demonstrates the loop:

```python
prefix = torch.tensor([[BOS, 3]])
model.eval()
with torch.inference_mode():
    for _ in range(4):
        next_scores = model(prefix)[:, -1, :].clone()
        next_scores[:, [PAD, BOS]] = float("-inf")
        next_id = next_scores.argmax(dim=-1, keepdim=True)  # greedy
        prefix = torch.cat((prefix, next_id), dim=1)
        if next_id.item() == EOS:
            break
```

Greedy decoding chooses the highest score. Sampling draws from `softmax(logits / temperature)` with a strictly positive temperature; smaller values concentrate the distribution and larger values flatten it. Positive temperature scaling does not change an argmax, so it has no diversity effect in an otherwise greedy loop. Top-k retains a number of candidates; top-p retains a probability-mass set. Beam search tracks multiple candidate sequences. These settings change outputs and cost without guaranteeing quality. [Hugging Face generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies).

With a pretrained causal language model, use its matching tokenizer and task class, pass the tokenizer's attention mask and use `max_new_tokens` when specifying an added-token budget. Respect its context limit, padding requirements and chat template. Library generation methods usually manage key/value caches to avoid repeating earlier projection work; the simple loop above recomputes the full prefix. A cache consumes additional memory and is runtime state, not a replacement for model weights. Benchmark prompt processing and per-token generation separately in [optimization and serving](../deployment/optimization-serving.md).

## Encoder-decoder cross-attention reads a separate source

Translation has two sequences: source IDs `[N,S]` and target IDs `[N,T]`. The encoder creates contextual source representations **memory** `[N,S,D]`. Here “memory” means an activation tensor representing this input; it does not mean saved weights, long-term knowledge storage or a training checkpoint.

The decoder's masked self-attention uses target-derived queries/keys/values. Its cross-attention uses **target queries** and **source memory keys/values**, producing weights `[N,H,T,S]`. The output length follows the target query length. A target position can normally attend to every real source token; the target causal triangle should not be copied onto source cross-attention merely because both modules use attention.

<div class="recall-flow" role="group" aria-label="Encoder-decoder source and target paths">
<div><b>Source</b><code>source IDs [N,S]</code><small>source embedding and positions</small></div>
<div><b>Encoder</b><code>memory [N,S,D]</code><small>plus source padding mask [N,S]</small></div>
<div><b>Target prefix</b><code>BOS + earlier target IDs</code><small>target embedding and causal self-attention</small></div>
<div><b>Cross-attend</b><code>Q: target; K,V: memory</code><small>ignore padded source keys</small></div>
<div><b>Predict</b><code>target logits [N,T,V_target]</code><small>next target token, until EOS</small></div>
</div>
<p class="visual-caption">Conceptual sequence transformation. Source and target vocabularies, lengths and padding masks are independent contracts.</p>

**Excerpt — existing encoder/decoder modules, embedded inputs and masks:**

```python
# src_vectors: [N,S,D]; tgt_vectors: [N,T,D], each includes positions
# src_pad: [N,S]; tgt_pad: [N,T]; True means PAD
# target_blocked: [T,T]; True means a future target key
memory = encoder(src_vectors, src_key_padding_mask=src_pad)
decoded = decoder(
    tgt_vectors, memory,
    tgt_mask=target_blocked,
    tgt_key_padding_mask=tgt_pad,
    memory_key_padding_mask=src_pad,
)
logits = target_head(decoded)  # [N,T,V_target]
```

This is the contract used by `nn.TransformerDecoder`; an optional `memory_mask` has target/source shape `[T,S]` and restricts cross-attention only when the task requires it. [TransformerDecoder API](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html).

For teacher forcing, start the decoder with `BOS` followed by the known target prefix and score the target shifted by one, including EOS and excluding PAD. During generation, encode the source once and reuse memory while growing the target prefix. Mask source padding in both the encoder and the decoder's cross-attention. Using `tgt_pad` as `memory_key_padding_mask` is wrong even if source/target lengths happen to match.

Keep source and target tokenizers/vocabularies with their proper heads. Evaluate held-out source/target pairs, inspect omissions and added content, and choose task measures alongside qualitative checks. Training loss and plausible examples do not establish reliable translation or summarization.

## Remember and apply

**Check yourself:** which dimension gets softmax in `[N,H,T,S]`? The source/key dimension `S`. Does `True` always block attention? No; SDPA uses the reverse boolean convention. Why allow the causal diagonal? The current input predicts the next target. Is transformer “memory” the model's weights? No; it is the encoded source activation for this input.

Use [model lifecycle](../deployment/model-lifecycle.md) next to save weights together with architecture and tokenizer contracts, then [optimization and serving](../deployment/optimization-serving.md) to measure the model on the intended CPU or other deployment target.

**Continue reading:** [6.1 · Memory, checkpoints and tracking](../deployment/model-lifecycle.md). Return to [the learning map](../../index.md#choose-a-guide) to choose another topic.
