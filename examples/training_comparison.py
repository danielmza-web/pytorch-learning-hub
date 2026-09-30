"""Compare two learning rates fairly; original Course 2 recall experiment."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from recall_patterns import train_accumulated


def metrics(model, loader):
    model.eval()
    matrix = torch.zeros(3, 3, dtype=torch.long)
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            predictions = logits.argmax(1)
            matrix += torch.bincount(y * 3 + predictions, minlength=9).reshape(3, 3)
    tp = matrix.diag().float()
    precision = tp / matrix.sum(0).clamp_min(1)
    recall = tp / matrix.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "loss": total_loss / matrix.sum().item(),
        "accuracy": tp.sum().item() / matrix.sum().item(),
        "macro_f1": f1.mean().item(),
        "recall_by_class": recall.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def run(output: Path, epochs=8):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    x = torch.randn(181, 6)
    # Imbalanced synthetic labels; class metrics show more than accuracy alone.
    y = torch.where(x[:, 0] > 0.8, 2, torch.where(x[:, 1] > 0.0, 1, 0))
    train = TensorDataset(x[:133], y[:133])
    validation = DataLoader(TensorDataset(x[133:], y[133:]), batch_size=17)
    template = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 3))
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
            train_accumulated(model, loader, optimizer, "cpu", microbatches=3)
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
        "best_initial_lr": min(trials, key=lambda t: t["history"][-1]["loss"])["initial_lr"],
        "trials": trials,
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for trial in trials:
        last = trial["history"][-1]
        print(f"initial_lr={trial['initial_lr']}: loss={last['loss']:.3f}, "
              f"accuracy={last['accuracy']:.3f}, macro_f1={last['macro_f1']:.3f}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/training-comparison"))
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    run(args.output, args.epochs)
