---
title: Training — efficient pipelines
tags: [dataloader, profiling, lightning, mixed-precision]
last_reviewed: 2026-10-01
---

# Training — efficient pipelines

Keep the learning objective fixed while reducing wasted loading time, computation and memory. First locate the bottleneck; then change the setting that can affect it.
{ .page-lead }

```text
storage → decode / transform on CPU → batch → transfer → GPU work → update
              DataLoader workers                 model and optimizer
```

If the GPU waits for data, a bigger model is unlikely to help. If convolution dominates time, adding loader workers may change very little. Start from [a trustworthy validation comparison](training-quality.md), inspect preparation, transfers and computation separately.



**Code key:** “Runnable toy” includes imports and inputs. Other snippets are excerpts: reuse `torch`, `nn`, and the model, loader, tokenizer or helper named in the section. Projects contain the complete runnable scripts.

## DataLoader settings

<div class="recall-flow" role="group" aria-label="Input to output">
<div><b>Prepare on CPU</b><code>read → decode → transform</code><small>workers and prefetch keep batches ready</small></div>
<div><b>Transfer</b><code>CPU → device</code><small>pinned memory may help CUDA transfers</small></div>
<div><b>Compute</b><code>forward → backward → update</code><small>measure without profiler overhead</small></div>
</div>
<p class="visual-caption">Illustration: shapes and operations, not measured model performance.</p>
<div class="memory-parts"><b>Training memory includes</b><span>Parameters and buffers</span><span>Activations</span><span>Gradients</span><span>Optimizer state</span></div>
<p class="visual-caption">Categories, not a to-scale memory chart. Frozen parameters still occupy storage and participate in the forward pass.</p>

| Parameter | What it controls | Trade-off |
| --- | --- | --- |
| `batch_size` | samples loaded together | larger batches may improve throughput but use more memory |
| `num_workers` | CPU subprocesses loading/preparing samples | excessive workers add overhead and RAM pressure |
| `pin_memory` | page-locked host memory for transfers | useful to test for CUDA; costs host memory |
| `prefetch_factor` | batches prepared ahead **per worker** | raises queued data and RAM usage |
| `persistent_workers` | keep worker processes between epochs | reduces restart overhead but retains worker state |
| `shuffle` | training sample order | validation usually uses stable order |
| `drop_last` | discard an incomplete final batch | changes samples seen; sometimes useful for BatchNorm |
| `collate_fn` | how samples are assembled | needed for variable-length text or filtered samples |

An example configuration, not a recommended optimum:

```python
from torch.utils.data import DataLoader

workers = 2
options = {"prefetch_factor": 2, "persistent_workers": True} if workers else {}
loader = DataLoader(
    dataset, batch_size=64, shuffle=True, num_workers=workers,
    pin_memory=(device.type == "cuda"), **options,
)
for x, y in loader:
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    # forward, loss, backward, update
```

Here `dataset` and `device` are already defined. With two workers and a factor of two, roughly four batches can be prefetched, plus other active buffers. This is host-side preparation; it does not store the whole dataset on the GPU. `non_blocking=True` can reduce transfer synchronization; actual overlap depends on pinned memory, streams, hardware and the rest of the code. The model still runs on device tensors.

With `num_workers=0`, loading happens in the main process: omit `prefetch_factor` and keep `persistent_workers=False`. On Windows, define datasets/collators at module scope and put script launch/training under `if __name__ == "__main__":`. Start debugging with zero workers to expose the underlying exception. Worker-local counters and logs are not automatically aggregated into the parent dataset.

**Experiment order:** measure a baseline → try a few worker counts → sweep batch size → test pinned transfers → adjust prefetching. Record samples/sec and epoch time, host/GPU memory and validation quality. Loader startup and cached data can distort a one-off timing. The largest legal batch is not always fastest or best.

## What Lightning automates

Lightning organizes the same PyTorch computation rather than changing what a neural network learns.

| Component / method | Responsibility |
| --- | --- |
| `LightningDataModule.prepare_data()` | download / prepare files, when needed |
| `setup(stage)` | build split datasets and transforms |
| `train_dataloader()` / `val_dataloader()` | return loaders |
| `LightningModule.forward()` | inference computation |
| `training_step()` | compute and **return the loss** |
| `validation_step()` | compute/log validation quantities |
| `configure_optimizers()` | return optimizer and optional schedule |
| `Trainer.fit()` | orchestrate devices, backward, updates, epochs and validation |
| `Callback` hooks | checkpointing, stopping and diagnostics |

In ordinary automatic optimization, do not duplicate `backward()` and `optimizer.step()` inside `training_step`. A small original model wrapper:

??? note "Implementation excerpt · requires the objects described above"
    ```python
    import lightning.pytorch as pl
    import torch
    from torch import nn

    class SmallClassifier(pl.LightningModule):
        def __init__(self, features=6, classes=3, lr=1e-3):
            super().__init__()
            self.save_hyperparameters()
            self.network = nn.Linear(features, classes)

        def forward(self, x):
            return self.network(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            loss = nn.functional.cross_entropy(self(x), y)
            self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            loss = nn.functional.cross_entropy(self(x), y)
            self.log("val_loss", loss, on_epoch=True, batch_size=len(y))

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
    ```


Given compatible loaders, `pl.Trainer(max_epochs=5, accelerator="auto", devices=1).fit(model, train_loader, val_loader)` runs it. Lightning and TorchMetrics are optional ecosystem dependencies; the four existing runnable projects use ordinary PyTorch.

`max_epochs` limits passes over training data; `max_steps` limits optimizer steps. `accelerator` selects a device type and `devices` its count. `precision` selects the arithmetic policy; `accumulate_grad_batches` changes update frequency; `gradient_clip_val` limits gradients; `log_every_n_steps` controls logging frequency. `limit_train_batches` / `limit_val_batches` can shorten a debugging run, so record those limits when interpreting results. `pl.seed_everything(17, workers=True)` initializes seeds including loader workers; reproducibility still depends on operations, hardware and versions.

### Callbacks and scheduler configuration

Stopping ends a run; checkpointing saves weights. Configure each callback around the intended validation metric.

??? note "Code and details"
    ```python
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    stopping = EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=1e-3)
    saving = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(max_epochs=20, accelerator="auto", devices=1,
                         callbacks=[stopping, saving])
    ```

    `patience` counts validation checks without sufficient improvement. `stopping_threshold` ends a run when a target is reached; it is different from `min_delta`. Early stopping does not restore the best weights by itself: use the checkpoint's `best_model_path`. `fast_dev_run=True` tests a few batches, not model quality. `trainer.callback_metrics` contains logged metrics, and sanity-validation results should be excluded from experiment summaries.

    For a plateau schedule, return this structure from `configure_optimizers()`:

    ```python
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",  # match the logged name and mode="min"
            "interval": "epoch",
            "frequency": 1,
        },
    }
    ```

    `self.log` of a TorchMetrics object lets Lightning manage epoch state. If you manually manage metric objects outside that integration, use `update/compute/reset` and separate train/validation state.

## Profile a short representative run

A profiler records expensive operations and memory activity. It adds overhead, so use it to diagnose, then measure final speed without profiling.

```python
from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import schedule

profiler = PyTorchProfiler(
    dirpath="artifacts/profile", filename="training",
    schedule=schedule(wait=1, warmup=1, active=4, repeat=1),
    profile_memory=True, record_shapes=True,
)
trainer = pl.Trainer(max_steps=6, accelerator="auto", devices=1,
                     profiler=profiler, logger=False, enable_checkpointing=False)
# trainer.fit(lightning_model, datamodule=data_module)
```

`wait` skips collection; `warmup` prepares collection but discards its measurements; `active` records. Use enough representative batches to reach the active window.

Read **self time** (work attributed directly to an operation), **total time** (including nested calls), call counts and memory allocations. Do not sum nested total times as if they were independent. CPU dispatch time is not GPU execution time. A costly matrix operation suggests different remedies from time spent waiting for the loader.

If large channel counts dominate, test a smaller architecture and recheck validation. A structural speed gain is not evidence that the smaller model preserves quality.

## Mixed precision

Some operations can run in FP16/BF16 while sensitive work stays in FP32. This can reduce memory and accelerate supported hardware. It does not halve every stored tensor, and it does not guarantee an identical metric.

Lightning's `precision="16-mixed"` manages autocasting and gradient scaling; `"bf16-mixed"` uses BF16 on supported hardware; `"32-true"` is an FP32 baseline. Test precision separately before combining it with other changes.

CUDA FP16 in a manual training loop, with existing model/loader/optimizer:

```python
scaler = torch.amp.GradScaler("cuda")
model.train()
for x, y in loader:
    x, y = x.to("cuda"), y.to("cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = loss_fn(model(x), y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # needed before inspecting/clipping these gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

The model must already be on CUDA. Clipping is optional; choose it for a reason. Do not call `.half()` on everything when using autocast. The older `torch.cuda.amp.*` spellings appear in many tutorials; current equivalents live under `torch.amp` / `torch.autocast`. [Official AMP documentation](https://docs.pytorch.org/docs/stable/amp.html) explains the device and dtype requirements.

`torch.set_float32_matmul_precision("high")` is a separate FP32 matrix-multiplication trade-off on supported hardware; it does not enable AMP or change all tensor dtypes.

## Gradient accumulation

<div class="recall-flow" role="group" aria-label="Input to output">
<div><b>Microbatch 1</b><code>16 samples → backward</code><small>gradients accumulate; weights stay fixed</small></div>
<div><b>Microbatch 2</b><code>16 samples → backward</code><small>add to existing gradients</small></div>
<div><b>Microbatch 3</b><code>16 samples → backward</code><small>average by 48 actual samples</small></div>
<div><b>Update</b><code>optimizer.step()</code><small>one update, then clear gradients</small></div>
</div>
<p class="visual-caption">Single-device illustration. For 133 samples, full groups are 48 and 48; the final update uses 37.</p>

Accumulate gradients over several **microbatches**, then update once. On one device:

```text
effective batch = microbatch size × accumulation steps
32 × 4 = 128 samples per full update
```

<div class="interactive-panel" data-accumulation-lab>
  <div class="interactive-heading">Samples per update</div>
  <label>Microbatch <input type="number" min="1" value="32" data-microbatch></label>
  <label>Accumulation steps <input type="number" min="1" value="4" data-accumulation></label>
  <label>Samples per epoch <input type="number" min="1" value="1000" data-epoch-samples></label>
  <output data-accumulation-output aria-live="polite">32 × 4 = 128 samples per full update; 8 updates for 1000 samples.</output>
  <small>One device; no dropped samples. Arithmetic illustration, not a speed prediction.</small>
</div>

Lightning: `Trainer(accumulate_grad_batches=4)`. In manual PyTorch, gradients must be cleared at **update boundaries**, not every microbatch. The following original helper averages by the actual number of samples in each group, including an incomplete final group:

```python
--8<-- "examples/recall_patterns.py:accumulation"
```

This code needs `import torch`; it uses unweighted cross-entropy with no ignored labels. For class-weighted loss, the reduction denominator is the sum of selected class weights, not simply sample count. When mixing AMP and accumulation, retain one scaling factor throughout the group and unscale/step/update at its boundary.

Accumulation saves activation memory relative to physically processing the full effective batch. It may increase runtime. It also **does not recreate large-batch BatchNorm statistics**: BatchNorm still sees each microbatch. Dropout and floating-point order can differ. Account for fewer optimizer updates when configuring schedules.

## Measure latency and memory

Parameter count alone does not measure speed or peak training memory:

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
stored_bytes = sum(t.numel() * t.element_size()
                   for t in list(model.parameters()) + list(model.buffers()))
stored_mib = stored_bytes / 1024**2
```

The second quantity includes frozen weights and buffers, but excludes activations, gradients, optimizer state and allocator overhead. It is not a complete memory budget or serialized file size.

For a forward-only latency measurement:

??? note "Implementation excerpt · requires the objects described above"
    ```python
    import time

    model.eval()
    device = next(model.parameters()).device
    sample = sample.to(device)  # keep transfer outside this forward-only timing
    with torch.inference_mode():
        for _ in range(10):
            model(sample)  # warm up
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(50):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = 1000 * (time.perf_counter() - start) / 50
    ```


CUDA work is asynchronous; synchronization includes completion rather than just dispatch. State input shape, batch size, device, precision, warmup and repetitions. Batch-one latency and large-batch throughput are different. Measure end-to-end latency separately if decoding and transfers matter.

For CUDA peak tensor memory, call `torch.cuda.reset_peak_memory_stats(device)` before the measured run and `torch.cuda.max_memory_allocated(device) / 1024**2` after it. This is allocated tensor memory, not all memory shown by a system monitor. `empty_cache()` releases unused cached blocks; it cannot free live tensors or their graphs.

## Remember and apply

**Check yourself:** using accumulation halves the microbatch and preserves the effective batch. Should training become twice as fast? No: there are more forward/backward calls, and BatchNorm still uses the smaller batch. Measure time, memory and validation together.

Next, connect data variation with [image augmentation](../vision/augmentation.md), or see why [dynamic text padding](../text/tokens-embeddings.md#padding-truncation-and-attention-masks) is another efficiency decision.

**Sources:** [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html), [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html), [Lightning optimization](https://lightning.ai/docs/pytorch/stable/common/optimization.html), [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html).
