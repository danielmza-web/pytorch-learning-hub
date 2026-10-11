---
title: P8 · Model lifecycle and pruning
study_context: Course 3 model deployment workflows
tags: [project, deployment, checkpoints, pruning]
project_id: P8
last_reviewed: 2026-10-10
---

# P8 · Model lifecycle and pruning {#project-model-lifecycle-on-cpu}

Follow one tiny model from a live object in RAM to saved weights, a resumable checkpoint, a reusable encoder and a pruned state. Every comparison runs offline on CPU.
{ .page-lead }

<div class="learning-context" data-project-id="P8" markdown="1">

**P8 · Model lifecycle and pruning**
{ .learning-location }

**You will practice:** Restore weights, resume one update, reuse an encoder and inspect pruning state.

**Read first:** [6.1 · Memory, checkpoints and tracking](../guides/deployment/model-lifecycle.md) · [6.2 · Compression, export and deployment](../guides/deployment/optimization-serving.md)

**Data and setup:** Six generated feature rows and a tiny model; offline CPU, no downloads.

</div>

## What this experiment answers

Constructing `TinyClassifier()` allocates parameter tensors and defines how the encoder and head connect. Training changes those tensors in memory. Saving a `state_dict` writes tensor values and names to disk; restoring still needs compatible architecture code. The example performs one synthetic training step so the checkpoint contains actual optimizer momentum rather than an empty optimizer state.

The encoder maps four input features to eight hidden features. The head maps those eight features to two logits. Six synthetic rows keep the mechanism easy to inspect; no accuracy, compression speedup or real-world usefulness is claimed.

<div class="recall-flow" role="group" aria-label="A model from memory to a restored CPU object">
<div><b>Architecture</b><code>4 → 8 → 2</code><small>encoder plus classifier head</small></div>
<div><b>Live tensors</b><code>58 float32 values</code><small>training changes RAM state</small></div>
<div><b>Save</b><code>state_dict + contract</code><small>disk artifact persists</small></div>
<div><b>Reconstruct</b><code>TinyClassifier(**config)</code><small>compatible code allocates tensors</small></div>
<div><b>Restore</b><code>load_state_dict(...)</code><small>equal evaluation logits</small></div>
</div>
<p class="visual-caption">Conceptual lifecycle with this example's exact tensor dimensions; not measured performance.</p>

CPU means central processing unit; RAM is the process’s working memory. The encoder weight `[8,4]` maps four input features to eight outputs; its dense slots survive pruning. **Common trap:** `weights_only=True` restricts deserialization rather than selecting only the model entry. Inspect equality and file-byte fields separately from quality or speed claims.

## Choose what to save

<!-- learning-overview:artifacts -->
<section class="learning-overview" data-overview="artifacts" aria-label="Choose the artifact for the next task"><p><strong>Choose the artifact for the next task</strong></p><div class="overview-grid"><div class="overview-card"><b>Inference artifact</b><p><b>Save:</b> model parameters + persistent buffers, architecture/configuration and input contract.</p><p><b>Restore:</b> construct compatible model → load state → eval + no_grad → predict.</p></div><div class="overview-card"><b>Training checkpoint</b><p><b>Save:</b> model + optimizer + run state, and the input contract. Include scheduler/scaler/random/data state when needed.</p><p><b>Restore:</b> construct model and optimizer → restore their state → resume training deliberately.</p></div><div class="overview-card"><b>Encoder only</b><p><b>Save:</b> encoder state, configuration and feature/input contract.</p><p><b>Restore:</b> construct encoder → restore it → attach a separate new head → train or adapt that head.</p></div></div><p><small>State dictionaries do not contain the forward implementation. Optimizer moments are not restored by model weights alone.</small></p></section>
<!-- /learning-overview -->

| Artifact | Contents | What it enables |
| --- | --- | --- |
| `weights.pt` | Full model state dictionary | Reconstruct compatible architecture and run inference. |
| `checkpoint.pt` | Model and optimizer states, step count, configuration, labels and input contract | Restore this demonstration's next training update. |
| `encoder.pt` | Only the encoder state | Reuse the representation with a separately chosen head. |
| `pruned-folded.pt` | Full model state after folding a pruning mask | Restore zeroed dense weights without pruning hooks. |
| `report.json` | Equality checks, tensor shapes, zero counts and measured file sizes | Inspect evidence independently of any model-quality claim. |

An application checkpoint may also need scheduler/scaler states, random-number generator (RNG) state, dataset order or sampler position and library versions. This demonstration repeats the same CPU batch and optimizer configuration; it does not promise exact resumption of arbitrary training runs.

**Code key:** the short snippets are excerpts from the maintained script, not standalone programs. The complete source below supplies imports, classes, data and helper functions; run that script from the repository root.

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model = TinyClassifier(**checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
with torch.inference_mode():
    logits = model(inputs)
```

`map_location="cpu"` maps loaded tensor storage to CPU. `eval()` selects evaluation behavior for layers such as Dropout and BatchNorm; `inference_mode()` disables autograd bookkeeping. These have different jobs.

**`weights_only=True` does not discard metadata or optimizer entries.** It restricts which objects the unpickler may reconstruct. Our tensor dictionaries and basic metadata still load; selecting `checkpoint["model_state_dict"]` is a separate operation. A weights file also does not contain Python architecture code. Load artifacts from trusted sources; restricted loading is not a guarantee that arbitrary files are harmless. [PyTorch serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html).

## A snapshot must stop changing

```python
live_state = model.state_dict()
snapshot = deepcopy(live_state)
```

`live_state` is a new mapping whose tensors normally reference the model's storage. A later in-place training update can therefore change its values. `deepcopy` gives the chosen state its own tensor storage. The script deliberately changes the head, checks that the live mapping follows the update while the snapshot stays fixed, and then restores the snapshot. During real training, take this snapshot when validation improves; this example does not select a “best” model from validation data. [Saving and loading tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

## Reuse only the encoder

```python
torch.save(model.encoder.state_dict(), encoder_path)
fresh_encoder = TinyClassifier(**CONFIG).encoder
fresh_encoder.load_state_dict(
    torch.load(encoder_path, map_location="cpu", weights_only=True)
)
```

The submodule dictionary contains `0.weight` and `0.bias`; the full model uses `encoder.0.weight` and `encoder.0.bias`. Both describe the same layer in different namespaces. Loading the encoder into a compatible encoder produces identical features. The classifier head is separate and must be trained or loaded for its intended task. `strict=False` can tolerate missing/unexpected keys but does not make incompatible shapes match; inspect returned keys when intentionally loading partial full-model state.

## Pruning zeros is different from shrinking a tensor

The encoder weight has shape `[8, 4]`, containing 32 float32 values. `l1_unstructured(..., amount=0.5)` masks the 16 smallest absolute values. Before removal, the state contains `weight_orig` plus `weight_mask`. `prune.remove` folds their product into a regular `weight`, removes the mask/hook machinery and preserves the effective forward computation.

| Stage | Encoder shape | Zero values | State representation |
| --- | --- | --- | --- |
| Before pruning | `[8, 4]` | Seed-dependent | Dense `weight` |
| Mask active | `[8, 4]` | 16 in this seeded toy | `weight_orig` + `weight_mask` |
| Mask folded | `[8, 4]` | 16 | Dense `weight` |

The folded model still has 58 float32 values: **232 bytes of dense tensor payload**. Ordinary dense storage keeps slots for zeros. Folding does not make the weight tensor smaller, and this script measures no inference acceleration. Smaller architectures, quantized representation or sparse kernels address different deployment constraints. [Official pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html).

## Run and read the evidence

```bash
python examples/model_lifecycle.py
python examples/model_lifecycle.py --output artifacts/my-lifecycle-run
python tests/validate_model_lifecycle.py
```

No datasets, pretrained weights, GPU or network connection are needed. The output directory defaults to ignored `artifacts/model-lifecycle/`. The tests use temporary directories and leave maintained teaching assets untouched.

Inspect these report fields:

- `restored_max_logit_difference` and `weights_only_file_max_logit_difference`: zero means the checkpoint and weights paths reproduced the saved evaluation outputs.
- `encoder_max_feature_difference`: zero means the independently restored submodule reproduced its hidden features.
- `resumed_next_step_max_parameter_difference`: zero means restored momentum and model state reproduced this example's next update.
- `deepcopy_snapshot_unchanged` and `alias_followed_update`: both should be `true`, exposing the difference between a reference and an independent snapshot.
- `pruning`: compare unchanged shapes, zero count, state keys and equal outputs before/after mask folding.
- `sizes`: compare summed dense tensor payload with actual serialized file bytes. Their difference includes archive/container metadata and overhead. File bytes depend on serialization and are not a RAM estimate or latency benchmark.

The resumed update changes the live model later in the script; `weights.pt` remains the saved state after the first step. The pruning branch starts from that saved state, so these comparisons do not accidentally mix different training moments.

**Try changing:** replace the hidden width `8` with `4` and rerun. Check the parameter count and dense payload. Then compare that actual architecture reduction with pruning half the original weights. For a real deployment, measure quality and latency on representative inputs and the intended hardware before choosing either approach.

Continue with [model memory, state and restoration](../guides/deployment/model-lifecycle.md), then [optimization and serving](../guides/deployment/optimization-serving.md).

## Complete source

??? note "Open the maintained runnable script"
    ```python
    --8<-- "examples/model_lifecycle.py"
    ```

**Inspect after running:** Inspect equality checks, shapes, zero counts and actual file bytes in report.json; the script does not benchmark speed.

**Connect the result:** [6.1 · Memory, checkpoints and tracking](../guides/deployment/model-lifecycle.md) · [6.2 · Compression, export and deployment](../guides/deployment/optimization-serving.md). [Choose another project](index.md).
