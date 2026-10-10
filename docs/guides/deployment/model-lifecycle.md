---
title: 6.1 · Memory, checkpoints and tracking
tags:
  - deployment
  - serialization
guide_id: "6.1"
last_reviewed: 2026-10-10
---

# 6.1 · Model memory, checkpoints and tracking {#model-memory-checkpoints-and-tracking}

A trained model is a computation with learned numbers. To use it tomorrow, preserve those numbers and the information needed to interpret them. To continue training, preserve the update process too.
{ .page-lead }

<div class="learning-context" data-guide-id="6.1" markdown="1">

**Guide 6.1 · Save and deploy models**
{ .learning-location }

**You will learn:** Choose what to retain for inference, component reuse or continuing training.

**Before you start:** Read [1.1 · Build and train](../fundamentals/core-workflow.md); [3.2 · Pretrained models](../vision/pretrained-models.md) explains backbone/head reuse.

**Practice:** [P8 · Model lifecycle and pruning](../../projects/model-lifecycle.md)

</div>

**Terms:** random-access memory (RAM) holds live tensors while the process runs. A persistent buffer is saved module state that is not an optimizer-trained parameter. A checkpoint is a saved training state; inference uses a model to predict. FP32/FP16/INT8 use 32-bit float, 16-bit float and 8-bit integer representations. A mebibyte (MiB) is 1,048,576 bytes; a decimal megabyte (MB) is 1,000,000 bytes.

The same lifecycle applies to CNNs, embeddings and transformers. [2.2 · Efficient training](../training/efficient-training.md) explains runtime memory; this page explains what persists after a run.

**Code key:** self-contained toys declare their imports and inputs. Other snippets are implementation excerpts; use the model, loader or tokenizer named in the section. Optional packages and downloads are stated beside their code.

## From code to memory to disk

Calling `model = MyModel(...)` builds Python modules and initializes parameter tensors, normally in CPU RAM. `model.to("cuda")` moves parameter and buffer tensors to GPU memory; Python orchestration still runs on the host. The forward pass creates activations; training also needs saved values for backward, gradients and optimizer state. A file on disk is a persistent representation of selected state, not a running model.

<div class="recall-flow" role="group" aria-label="Model lifecycle">
<div><b>Construct</b><code>architecture + initial tensors</code><small>Shapes and forward rules come from code.</small></div>
<div><b>Learn in memory</b><code>inputs → loss → updates</code><small>Weights change; activations and gradients occupy memory.</small></div>
<div><b>Serialize to disk</b><code>state_dict + metadata</code><small>Persist learned tensors and their input/output contract.</small></div>
<div><b>Reconstruct and infer</b><code>same architecture + saved state</code><small>Restore, set evaluation mode and supply matching inputs.</small></div>
</div>

`state_dict()` maps names to parameters **and persistent buffers**, such as BatchNorm running statistics. It does not contain the `forward` implementation or all Python attributes. An optimizer has its own state dictionary. Sources: [module state](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict), [serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html).

## What occupies memory?

For a dense tensor, payload bytes are `numel() * element_size()`. Ten million FP32 parameters need about 40 MB (decimal), or 38.15 MiB, before buffers and runtime overhead. FP16 uses two bytes per stored value and INT8 one, but quantized formats also store scales, zero points and sometimes packing metadata. Converting some layers does not change every tensor.

<div class="interactive-panel" data-weight-budget>
<h3>Change the weight-storage budget</h3>
<label>Parameter count <input data-parameters type="number" min="1" max="1000000000000" step="1" value="10000000"></label>
<label>Stored bits per weight <select data-weight-bits><option value="32">FP32 · 32 bits</option><option value="16">FP16/BF16 · 16 bits</option><option value="8">INT8 · 8 bits</option></select></label>
<div data-budget-bars></div><output data-budget-output aria-live="polite">Enable JavaScript to compare payload estimates.</output>
<small>Exact arithmetic for hypothetical dense weight payloads. Excludes activations, buffers, gradients, optimizer state, quantization metadata and runtime overhead; this is not a device benchmark.</small>
</div>

FP32 training with Adam commonly has weights, gradients and two moment tensors: roughly 16 bytes per parameter just for those four components. This is an estimate, not peak memory: activations depend on batch/sequence/image size, optimizer initialization can be lazy, mixed precision can retain additional copies, and allocators reserve extra memory. Inference avoids optimizer state and backward graphs; it still needs weights, activations and temporary workspaces. [Adam state](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) and [CUDA memory](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management) describe the implementation limits.

## Choose what to save

<div class="learning-comparison" role="group" aria-label="Live memory, an inference artifact and a resume checkpoint">
<div><b>Live process</b><code>architecture + tensors in RAM</code><small>Training changes the live values; closing the process loses unsaved state.</small></div>
<div><b>Inference artifact</b><code>model state + input contract</code><small>Reconstruct compatible architecture, restore state and predict.</small></div>
<div><b>Resume checkpoint</b><code>model + optimizer + run state</code><small>Also restore the update process and any scheduler/scaler/data position needed.</small></div>
</div>
<p class="visual-caption">Conceptual storage choices. A model state dictionary includes persistent buffers; file size and peak process memory are different quantities.</p>

| Artifact | What belongs in it | Use |
| --- | --- | --- |
| Weights for inference | model state plus architecture/configuration, class order, normalization or tokenizer | reproduce predictions |
| Training checkpoint | model, optimizer, epoch/step, scheduler, AMP scaler if used, random state and data/sampler position when needed | resume learning |
| Backbone or encoder | that submodule's state plus its configuration and expected feature shape | reuse learned features |
| Exported graph | supported inference computation and tensors, with shape constraints | another compatible runtime |

Saving the entire module with `torch.save(model, ...)` couples restoration to Python class definitions and pickle behavior. A state dictionary is the usual portable choice. Extensions `.pt` and `.pth` are conventions, not different mathematical formats. Neither a filename nor `weights_only=True` tells you what the creator actually placed inside the file. [Saving/loading tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

### Save and reload weights

This self-contained CPU example demonstrates restoration, not model quality. Replace the small architecture and metadata together in a real application.

```python
from pathlib import Path
import torch
from torch import nn

torch.manual_seed(7)
path = Path("artifacts/lifecycle")
path.mkdir(parents=True, exist_ok=True)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
model.eval()
x = torch.randn(3, 4)
with torch.inference_mode():
    before = model(x)

torch.save({"model": model.state_dict(),
            "classes": ["clear", "inspect"],
            "input_features": 4}, path / "weights.pt")
saved = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
restored = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
restored.load_state_dict(saved["model"])
restored.eval()
with torch.inference_mode():
    torch.testing.assert_close(before, restored(x))
```

`map_location="cpu"` allows CPU restoration of tensors saved from a GPU. `weights_only=True` restricts deserialization to supported tensor/primitive structures; it does **not** discard optimizer tensors or automatically extract a `model` entry. Load only files from trusted sources; restricted loading is not a general validation of file contents. [torch.load](https://docs.pytorch.org/docs/stable/generated/torch.load.html).

`eval()` changes Dropout and BatchNorm behavior. `inference_mode()` avoids autograd recording and additional tracking overhead; they serve different purposes. Saliency requires gradients even when the model is in evaluation mode, so use ordinary grad-enabled execution for that task.

### Save only a component

A classifier may be `encoder → head`. If the next task needs only features, save `model.encoder.state_dict()` and later construct the same encoder before loading. The head can be discarded or replaced. Save buffers and preprocessing with the component; features from incompatible preprocessing are not equivalent.

??? note "Component state and compatible partial loading"
    ```python
    # model.encoder and new_encoder must have the same architecture.
    torch.save(model.encoder.state_dict(), "encoder.pt")
    encoder_state = torch.load("encoder.pt", map_location="cpu", weights_only=True)
    new_encoder.load_state_dict(encoder_state)
    ```

    A whole-model state uses names such as `encoder.0.weight`. A separately saved encoder uses local names such as `0.weight`. Match the destination and naming scheme. `strict=False` tolerates missing/unexpected names, but does not make same-name tensors with different shapes compatible. Filter intentional mismatches and inspect the returned missing/unexpected keys; do not silently ignore an accidental incomplete load.

## Preserve the best state and resume deliberately

`best = model.state_dict()` keeps references to the live tensor storage. Later updates can change that apparent snapshot. Use `copy.deepcopy(model.state_dict())`, or detach/clone each tensor, when retaining a best state in memory. Saving immediately to disk creates a separate serialized snapshot. [Snapshot guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended).

??? note "What a resume checkpoint needs"
    ```python
    # Existing training objects; keep metadata to supported primitive types.
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": float(best_val_loss),
    }
    torch.save(checkpoint, "resume.pt")
    ```

    Reconstruct the model on the target device, then its optimizer/scheduler with the intended parameter groups; load all their states and return to `train()`. Check optimizer/scheduler initialization and loading order in their documentation. Restore scaler/random-generator/data position state when the application needs them. Resuming at an epoch boundary is simpler than replaying the exact next minibatch. A seed alone does not guarantee exact continuation across software versions, devices or loader workers. [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

### Load large models without two ordinary allocations

For supported PyTorch checkpoint formats, `torch.load(..., mmap=True, map_location="cpu", weights_only=True)` maps storage for demand-driven access. Constructing modules inside `with torch.device("meta"):` creates tensor descriptions without ordinary data storage; `load_state_dict(state, assign=True)` can then attach loaded tensors. Create the optimizer after assignment so it references the actual parameters. These techniques reduce loading allocation pressure; inference still needs accessible weights and workspaces. An in-memory dictionary filtered **after** `torch.load` is not selective disk loading. [Loading recipe](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html).

## Track the experiment and the deployable artifact

MLflow connects a run ID to settings, step-indexed metrics and artifacts. Record the code revision, split/data version, class/tokenizer mapping, preprocessing, model configuration, software versions and device. Log validation metrics during selection; keep final test evaluation separate. A tracking run is not a deployed endpoint, and a registry entry does not prove a candidate is ready.

??? note "Optional local MLflow tracking"
    Requires the separately installed `mlflow` package. It is not needed for the offline lifecycle project.

    ```python
    import mlflow
    mlflow.set_tracking_uri("file:./artifacts/mlruns")
    mlflow.set_experiment("inspection-baseline")
    with mlflow.start_run():
        mlflow.log_params({"hidden": 8, "split_seed": 7})
        mlflow.log_metric("val_loss", float(val_loss), step=epoch)
        mlflow.log_artifact("artifacts/lifecycle/weights.pt")
    ```

    A local artifact store needs no public service. Team tracking can use an authenticated server and external storage; configure those separately. Treat large model files as versioned artifacts and record their digest in source control, rather than putting datasets and every checkpoint into ordinary Git. [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/).

## Check your understanding

You close Python after training. The checkpoint survives, but the model object in RAM does not. To reproduce a prediction, reconstruct the architecture, restore state, use the saved preprocessing/class order and evaluate. To resume Adam, weights alone are insufficient because the optimizer's moments are missing.

Run [the original lifecycle project](../../projects/model-lifecycle.md) to test a weight round trip, encoder-only reuse, an independent snapshot and unchanged dense shapes after pruning. Then continue to [compression, export and deployment](optimization-serving.md).

**Continue reading:** [6.2 · Compression, export and deployment](optimization-serving.md). Return to [the learning map](../../index.md#choose-a-guide) to choose another topic.
