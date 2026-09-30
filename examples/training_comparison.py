"""Compare two learning rates fairly; original Course 2 recall experiment."""
from __future__ import annotations

import argparse
import copy
import json
import time
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from recall_patterns import train_accumulated


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; choose --device cpu")
    return torch.device(name)


def metrics(model, loader):
    model.eval()
    matrix = torch.zeros(3, 3, dtype=torch.long)
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            device = next(model.parameters(), torch.empty(0)).device
            logits = model(x.to(device))
            total_loss += nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
            predictions = logits.argmax(1).cpu()
            matrix += torch.bincount(y * 3 + predictions, minlength=9).reshape(3, 3)
    tp = matrix.diag().float()
    precision = tp / matrix.sum(0).clamp_min(1)
    recall = tp / matrix.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    count = matrix.sum().item()
    return {
        "loss": total_loss / count if count else None,
        "accuracy": tp.sum().item() / count if count else None,
        "macro_f1": f1.mean().item() if count else None,
        "recall_by_class": recall.tolist() if count else [None] * 3,
        "confusion_matrix": matrix.tolist(),
    }


def benchmark(template, train, device, repetitions=5):
    """Tiny end-to-end training epoch; fixed order, effective batch 48, three updates."""
    if repetitions < 1 or not len(train):
        raise ValueError("Benchmark requires samples and positive repetitions")
    variants = [("physical FP32", 48, 1, False), ("accumulated FP32", 16, 3, False)]
    if device.type == "cuda":
        variants.append(("accumulated AMP FP16", 16, 3, True))
    cuda_memory = None
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        cuda_memory = {"free_mib_before_trials": free / 1024**2, "total_mib": total / 1024**2,
                       "device_name": torch.cuda.get_device_name(device)}
    results = []
    for name, size, accumulation, amp in variants:
        elapsed, peaks, states = [], [], []
        for repetition in range(repetitions + 1):
            model = copy.deepcopy(template).to(device).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            loader = DataLoader(train, batch_size=size, shuffle=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            group = 0
            for step, (x, y) in enumerate(loader, 1):
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    loss = nn.functional.cross_entropy(model(x), y, reduction="sum")
                scaler.scale(loss).backward()
                group += len(y)
                if step % accumulation == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    for p in model.parameters():
                        p.grad.div_(group)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    group = 0
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            if repetition:  # discard the first complete epoch as warmup
                elapsed.append(seconds)
                if device.type == "cuda":
                    peaks.append(torch.cuda.max_memory_allocated(device) / 1024**2)
            states.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        median = statistics.median(elapsed)
        results.append({"name": name, "microbatch": size, "accumulation": accumulation,
                        "effective_batch": 48, "samples": len(train), "updates": (len(train) + 47) // 48,
                        "median_seconds": median, "seconds": elapsed,
                        "samples_per_second": len(train) / median,
                        "peak_cuda_allocated_mib": max(peaks) if peaks else None})
        if len(results) == 1:
            reference = states[-1]
        results[-1]["max_parameter_difference_from_physical_fp32"] = max(
            (states[-1][k] - reference[k]).abs().max().item() for k in reference)
    return {"device": str(device), "torch": str(torch.__version__), "threads": torch.get_num_threads(),
            "scope": "tiny training epoch including loading and transfer; not a hardware ranking",
            "warmup_epochs": 1, "repetitions": repetitions, "variants": results,
            "cuda_memory": cuda_memory,
            "optimizer": {"name": "SGD", "learning_rate": 0.1}, "model": str(template),
            "amp": "measured" if device.type == "cuda" else "skipped: CUDA required"}


def run(output: Path, epochs=8, device="cpu", do_benchmark=False):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    x = torch.randn(181, 6)
    # Imbalanced synthetic labels; class metrics show more than accuracy alone.
    y = torch.where(x[:, 0] > 0.8, 2, torch.where(x[:, 1] > 0.0, 1, 0))
    train = TensorDataset(x[:133], y[:133])
    validation = DataLoader(TensorDataset(x[133:], y[133:]), batch_size=17)
    device = device_for(device)
    template = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 3)).to(device)
    initial = copy.deepcopy(template.state_dict())
    trials = []
    for lr in (0.01, 0.1):
        model = copy.deepcopy(template)
        model.load_state_dict(initial)
        # Recreate the generator: each trial sees the same shuffled batch order.
        loader = DataLoader(train, batch_size=16, shuffle=True,
                            generator=torch.Generator().manual_seed(23))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1)
        history = []
        for epoch in range(epochs):
            used_lr = optimizer.param_groups[0]["lr"]
            # 3 x 16 = 48 samples/full update; final group uses its actual count.
            train_accumulated(model, loader, optimizer, device, microbatches=3)
            measured = metrics(model, validation)
            scheduler.step(measured["loss"])
            history.append({"epoch": epoch + 1, "lr_used": used_lr,
                            "lr_next": optimizer.param_groups[0]["lr"], **measured})
        trials.append({"initial_lr": lr, "history": history})
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "data": "seeded synthetic classification; not a real-world benchmark",
        "selection": "lowest final validation loss; no test set used for tuning",
        "train_samples": len(train), "validation_samples": 48,
        "microbatch": 16, "accumulation_steps": 3, "seed": 17,
        "device": str(device), "torch": str(torch.__version__),
        "best_initial_lr": min(trials, key=lambda t: t["history"][-1]["loss"])["initial_lr"],
        "trials": trials,
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if do_benchmark:
        (output / "benchmark.json").write_text(json.dumps(benchmark(template, train, device), indent=2), encoding="utf-8")
    for trial in trials:
        last = trial["history"][-1]
        print(f"initial_lr={trial['initial_lr']}: loss={last['loss']:.3f}, "
              f"accuracy={last['accuracy']:.3f}, macro_f1={last['macro_f1']:.3f}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/training-comparison"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--benchmark", action="store_true", help="Measure a warmed-up physical/accumulated comparison")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    run(args.output, args.epochs, args.device, args.benchmark)
