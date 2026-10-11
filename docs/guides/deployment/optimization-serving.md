---
title: 6.2 · Compression, export and deployment
tags:
  - deployment
  - quantization
guide_id: "6.2"
last_reviewed: 2026-10-11
---

# 6.2 · Compression, export and deployment {#compression-export-and-deployment}

A model that works on a training computer still needs to fit the target's memory, latency and quality budget. Preserve a tested baseline, change one mechanism at a time, and measure the complete prediction path on the device that will run it.
{ .page-lead }

<div class="learning-context" data-guide-id="6.2" markdown="1">

**Guide 6.2 · Save and deploy models**
{ .learning-location }

**You will learn:** Distinguish smaller tensors from fewer bits, then validate a candidate on its deployment target.

**Before you start:** Read [6.1 · Memory, checkpoints and tracking](model-lifecycle.md) so the baseline and its input contract can be restored.

**Practice:** [P8 · Model lifecycle and pruning](../../projects/model-lifecycle.md)

</div>

**Terms:** pruning removes or masks selected connections; quantization changes numerical representation; distillation trains a smaller student. INT8 means 8-bit integers; FP32 means 32-bit float. Quantization-aware training (QAT) simulates conversion effects during training. Open Neural Network Exchange (ONNX) is a model representation for compatible runtimes, not a running service.

[Efficient architectures](../architectures/vision-generative.md) reduce work by design; this page explains what to do with a model you already trained.

**Code key:** self-contained toys declare their imports and inputs. Other snippets are implementation excerpts; use the model, loader or tokenizer named in the section. Optional packages and downloads are stated beside their code.

## Choose a mechanism for the actual bottleneck

<div class="recall-flow" role="group" aria-label="Deployment decisions">
<div><b>Baseline</b><code>quality + bytes + latency</code><small>Measure the unchanged model and preprocessing.</small></div>
<div><b>Change</b><code>smaller architecture / compression</code><small>Choose a mechanism the runtime supports.</small></div>
<div><b>Validate</b><code>same inputs and held-out data</code><small>Check numerical drift and task-level errors.</small></div>
<div><b>Deploy</b><code>versioned artifact + contract</code><small>Measure on the target, monitor and keep rollback.</small></div>
</div>

| Mechanism | What changes | What must be verified |
| --- | --- | --- |
| Smaller model / lower input resolution | channels, layers or spatial/sequence work | acceptable loss of detail or capacity |
| Pruning | selected connections or channels | whether the runtime exploits the resulting structure |
| Quantization | numerical representation and supported kernels | quality, calibration and target operator support |
| Distillation | a student learns from a larger teacher | new training and independent student evaluation |
| Export / graph optimization | computation representation, fusions and execution backend | supported operations, shapes and output parity |

Training on a GPU does not oblige you to infer on a GPU. Compatible weights can be loaded on CPU, but a large model may still be too slow or too large. Reducing **training** activation memory with gradient accumulation does not compress the inference model. [PyTorch CPU example](https://docs.pytorch.org/tutorials/intermediate/realtime_rpi.html).

## Pruning: zeros are not smaller tensor shapes

<!-- learning-overview:compression -->
<section class="learning-overview" data-overview="compression" aria-label="Zeros, fewer slots and fewer bits are different"><p><strong>Zeros, fewer slots and fewer bits are different</strong></p><div class="overview-grid"><div class="overview-card"><b>Mask half the entries</b><figure class="learning-matrix" tabindex="0" aria-label="8×4 FP32 · 32 slots. Arrow keys scroll a wide matrix."><figcaption>8×4 FP32 · 32 slots</figcaption><div class="learning-cells" style="--columns:4"><span>0</span><span>1</span><span>0</span><span>1</span><span>1</span><span>0</span><span>1</span><span>0</span><span>0</span><span>1</span><span>0</span><span>1</span><span>1</span><span>0</span><span>1</span><span>0</span><span>0</span><span>1</span><span>0</span><span>1</span><span>1</span><span>0</span><span>1</span><span>0</span><span>0</span><span>1</span><span>0</span><span>1</span><span>1</span><span>0</span><span>1</span><span>0</span></div></figure><p><strong>128 bytes</strong> · zeros still occupy dense slots.</p></div><div class="overview-card"><b>Actually smaller layer</b><figure class="learning-matrix" tabindex="0" aria-label="4×4 FP32 · 16 slots. Arrow keys scroll a wide matrix."><figcaption>4×4 FP32 · 16 slots</figcaption><div class="learning-cells" style="--columns:4"><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span></div></figure><p><strong>64 bytes</strong> · connected layers must also adapt.</p></div><div class="overview-card"><b>Hypothetical INT8 storage</b><figure class="learning-matrix" tabindex="0" aria-label="8×4 INT8 · 32 slots. Arrow keys scroll a wide matrix."><figcaption>8×4 INT8 · 32 slots</figcaption><div class="learning-cells" style="--columns:4"><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span><span>1</span></div></figure><p><strong>32 bytes</strong> · lower bit width needs a supported representation and kernels.</p></div></div><p><small>Dense payload illustrations only; quantization metadata is extra. No speed or accuracy prediction.</small></p></section>
<!-- /learning-overview -->


<div class="learning-comparison" role="group" aria-label="Zero values, fewer bits and fewer dense entries">
<div><b>Masked dense weights</b><code>8 × 4 FP32 → 128 bytes</code><small>32 stored slots remain even if 16 are zero.</small></div>
<div><b>Hypothetical INT8 weights</b><code>8 × 4 INT8 → 32 bytes</code><small>Same 32 slots, fewer bits per value; metadata and support matter.</small></div>
<div><b>Smaller dense layer</b><code>4 × 4 FP32 → 64 bytes</code><small>16 slots; connected layers must be adapted and the new model evaluated.</small></div>
</div>
<p class="visual-caption">Exact hypothetical weight payloads, excluding bias, metadata and other layers. Neither these byte counts nor a pruning mask establish a runtime speedup.</p>

Unstructured pruning chooses individual weights. Structured pruning masks groups such as output channels. Global pruning distributes a sparsity budget across chosen layers. A mask-based PyTorch operation normally retains the original dimensions: a dense matrix multiplication still processes the dense tensor unless the selected backend has a suitable sparse kernel.

```python
import torch
from torch import nn
from torch.nn.utils import prune

layer = nn.Linear(12, 6)
shape = layer.weight.shape
prune.l1_unstructured(layer, name="weight", amount=0.5)
# State contains weight_orig and weight_mask while parametrized.
prune.remove(layer, "weight")
assert layer.weight.shape == shape  # Still [6, 12].
assert (layer.weight == 0).sum().item() == 36
```

`remove` folds the mask into a normal parameter; it does not remove rows, columns or channels. Actually shrinking a network requires consistent changes to connected layers, normalization and shortcuts, or an explicitly supported sparse representation. Fine-tune after pruning as needed, then re-evaluate quality and measure speed. The [pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html) explains the mask/state representation. The [offline project](../../projects/model-lifecycle.md) checks it on a tiny CPU model.

## Quantization: fewer bits with a numerical trade-off

<!-- learning-overview:quantization -->
<section class="learning-overview" data-overview="quantization" aria-label="Rounding changes the reconstructed number"><p><strong>Rounding changes the reconstructed number</strong></p><figure class="overview-flow"><figcaption>Toy signed 4-bit range −8…7 · scale 0.25</figcaption><ol><li><b>Original</b><span>1.3</span></li><li><b>Divide + round</b><span>1.3 / 0.25 = 5.2 → 5 (ties to even).</span></li><li><b>Clip</b><span>5 is inside −8…7.</span></li><li><b>Reconstruct</b><span>5 × 0.25 = 1.25; error = -0.05.</span></li></ol></figure><p><small>This calculation uses 4 bits for visibility; it is separate from the INT8 payload comparison.</small></p></section>
<!-- /learning-overview -->

<details class="learning-exploration"><summary>Try a different value</summary>
<div data-learning-lab="quantization"><p>Fixed scale 0.25; signed 4-bit range −8…7. Enable JavaScript to change only the original value.</p></div>
</details>

A simple integer representation approximates `real ≈ scale * (integer - zero_point)`. Rounding and clipping introduce error. Weight-only quantization reduces weight storage while activations can remain floating point. Quantizing activations as well requires ranges and kernels that match the hardware. Model payload, actual serialized bytes and process peak memory are different measurements.

| Method | How activation ranges are obtained | Practical use |
| --- | --- | --- |
| Dynamic post-training quantization | calculated while running supported operations | useful for selected linear/recurrent workloads; runtime range computation has cost |
| Static post-training quantization | observers see representative calibration inputs before conversion | fixed deployment ranges for supported graph/backend combinations |
| Quantization-aware training (QAT) | training simulates rounding/clipping using fake quantization | adapt weights when post-training conversion loses too much quality |

Calibration normally needs representative **inputs**, not training updates or labels. Use a training-derived/calibration split, preserve preprocessing, and cover realistic ranges. Keep the final test set for final evaluation. Poor range coverage can clip rare but important inputs. QAT uses floating-point training with simulated quantization; only the final compatible conversion produces deployment operators. See [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

### Course APIs and the current implementation path

The local course uses eager APIs such as `torch.quantization.quantize_dynamic`, `prepare`, `convert`, `prepare_qat`, `QuantStub` and layer fusion. Understand that sequence, but do not assume every backend supports it. The current [PyTorch quantization documentation](https://docs.pytorch.org/docs/stable/quantization.html) directs new development toward [torchao](https://docs.pytorch.org/ao/stable/index.html); legacy eager/FX and export-based workflows have different support and preparation requirements. Select a tested version/backend combination before following an implementation tutorial.

An unsupported CNN may keep convolution layers in float while quantizing only its classifier. FP16 tensors do not automatically improve small-CPU latency. Four-times smaller INT8 **weight payload** relative to FP32 does not imply four-times lower application RAM or four-times faster inference. Quantization must be evaluated, not assumed to preserve accuracy.

## Distillation: a smaller student is a new model

A teacher provides soft class scores or intermediate features alongside ordinary labels. A student can learn relationships between classes that a hard label omits. A common objective combines supervised cross-entropy with a temperature-scaled KL divergence between teacher and student distributions; the teacher stays frozen in evaluation mode. The student needs its own architecture, training and validation. Distillation is an extension of the course's compression topics, not a substitute for measuring the trained student. [Official distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html).

## ONNX: export a supported computation

ONNX describes operations and tensor contracts for compatible runtimes. It is not automatically a speed improvement and does not transport arbitrary Python side effects. Export the evaluation model using representative inputs and declare which dimensions may vary. The current PyTorch exporter uses the `torch.export` path with `dynamo=True`; older course snippets use the legacy tracer. TorchScript is deprecated in current PyTorch documentation. [Exporter guide](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html), [TorchScript status](https://docs.pytorch.org/docs/stable/jit.html).

??? note "Optional original export and output comparison"
    Requires `onnx`, `onnxscript` and `onnxruntime` compatible with the installed PyTorch version. These are optional, not required for the website or offline lifecycle project. This example exports a fixed feature dimension with a dynamic batch dimension; sequence/image dimensions need their own constraints.

    ```python
    from pathlib import Path
    import numpy as np
    import torch
    from torch import nn
    import onnxruntime as ort

    torch.manual_seed(9)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
    folder = Path("artifacts/export")
    folder.mkdir(parents=True, exist_ok=True)
    sample = torch.randn(2, 4)
    torch.onnx.export(
        model, (sample,), folder / "model.onnx", dynamo=True,
        input_names=["features"], output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=16)},),
    )
    session = ort.InferenceSession(str(folder / "model.onnx"),
                                  providers=["CPUExecutionProvider"])
    for batch in (1, 3, 16):
        x = torch.randn(batch, 4)
        with torch.inference_mode():
            expected = model(x).numpy()
        actual = session.run(None, {"features": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    ```

    This checks arithmetic for an untrained toy network, not application quality. Inspect graph validity/operator support and test realistic values, allowed shapes and output ordering too. Large models may use an adjacent external weights file; keep the complete artifact together. A successful export alone does not verify the target runtime. [ONNX Runtime inference](https://onnxruntime.ai/docs/api/python/api_summary.html).

For compression, compare float and converted predictions on the same examples. Small logit drift can change a borderline class even when averages look similar. Repeat task evaluation with the real preprocessing and output interpretation, including rare classes or safety-critical operating ranges relevant to the application.

## Measure on the target device

Record serialized bytes, process/peak tensor memory, batch-one latency distribution and throughput under the actual workload. Warm up before timing, report cold start separately, use repeated measurements and record input shape, dtype, device, thread count and runtime/provider versions. Synchronize GPU work around a timer. Measure preprocessing, transfer, execution and postprocessing separately and together; a faster kernel may barely affect an application dominated by image decoding. [PyTorch benchmark utilities](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html).

For a small CPU, begin with a compact architecture and realistic resolution, test supported quantization and tune threads. For an embedded runtime, also check supported operations and total storage, boot time and peak RAM. [ExecuTorch](https://docs.pytorch.org/executorch/stable/index.html) is a current edge-deployment option beyond the course's ONNX workflow; its requirements depend on the backend. An ONNX file and a `.pt` file are not interchangeable inputs to every runtime.

## Package, monitor and roll back

Ship the artifact with a manifest: model/config version, input names/shapes/dtypes/ranges, resize/normalization or tokenizer version, label order, output interpretation, runtime dependencies, validation results and artifact digest. Test malformed inputs and expected boundary shapes. Load the model once for repeated inference; recreating it for each request adds latency.

Deployment can be a local process, a device application or a service. Exporting or logging a model is only preparation. In operation, monitor failures, latency, resource use and input drift; quality monitoring needs delayed labels or a defensible review process. Keep a previous validated artifact and its matching preprocessing for rollback. These are application decisions, not features of this static learning website.

## Check your understanding

You mask half of a dense FP32 matrix and save it. Its shape and value dtype remain the same, so do not claim a half-sized model or faster dense CPU inference. Quantization may reduce supported weight payloads; calibration, conversion, target timing and held-out task evaluation decide whether the result is useful. Return to [checkpoints](model-lifecycle.md) when deciding what to retain for retraining versus inference.

**Apply the route:** [P8 · Model lifecycle and pruning](../../projects/model-lifecycle.md), or [choose a project](../../projects/index.md). Return to [the learning map](../../index.md#choose-a-guide) for another topic.
