"""Original offline CPU experiment: trained state, checkpoints, parts and pruning.

This demonstrates mechanisms on synthetic data, not useful classifier quality.
Only checkpoints created by this example are loaded.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import prune


SEED = 71
CONFIG = {"input_features": 4, "hidden_features": 8, "classes": 2}


class TinyClassifier(nn.Module):
    """Architecture is code; a state_dict supplies parameter/buffer values."""

    def __init__(self, input_features=4, hidden_features=8, classes=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_features, hidden_features), nn.ReLU())
        self.head = nn.Linear(hidden_features, classes)

    def forward(self, inputs):
        return self.head(self.encoder(inputs))


def synthetic_batch():
    generator = torch.Generator().manual_seed(SEED + 1)
    inputs = torch.randn(6, CONFIG["input_features"], generator=generator)
    labels = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
    return inputs, labels


def train_step(model, optimizer, inputs, labels):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(inputs), labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def tensor_payload_bytes(value):
    """Sum dense tensor leaf payloads; exclude archive/container overhead.

    This is not process RAM usage or a general storage-deduplicating estimator.
    Our tiny state contains separate, dense tensors with no tied storage.
    """
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_payload_bytes(item) for item in value)
    return 0


def output_logits(model, inputs):
    model.eval()
    with torch.inference_mode():
        return model(inputs).clone()


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    model = TinyClassifier(**CONFIG).cpu()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    inputs, labels = synthetic_batch()
    train_step(model, optimizer, inputs, labels)
    expected_logits = output_logits(model, inputs)

    # A state_dict is a mapping, but its tensors normally share model storage.
    live_state = model.state_dict()
    snapshot = deepcopy(live_state)
    original_head = snapshot["head.weight"].clone()
    with torch.no_grad():
        model.head.weight.add_(1.0)  # stand-in for later training updates
    alias_followed_update = torch.equal(live_state["head.weight"], model.head.weight)
    snapshot_isolated = torch.equal(snapshot["head.weight"], original_head)
    model.load_state_dict(snapshot)  # restore the chosen state before saving

    weights_path = output / "weights.pt"
    checkpoint_path = output / "checkpoint.pt"
    encoder_path = output / "encoder.pt"
    torch.save(model.state_dict(), weights_path)
    checkpoint = {
        "model_state_dict": snapshot,
        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
        "completed_steps": 1,
        "config": CONFIG,
        "labels": ["class-zero", "class-one"],
        "input_contract": "six float32 rows, four synthetic features per row",
        "seed": SEED,
    }
    torch.save(checkpoint, checkpoint_path)
    torch.save(model.encoder.state_dict(), encoder_path)

    # weights_only restricts unpickling; it does not select dictionary fields.
    loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = TinyClassifier(**loaded_checkpoint["config"]).cpu()
    restored.load_state_dict(loaded_checkpoint["model_state_dict"])
    restored_logits = output_logits(restored, inputs)
    weights_restored = TinyClassifier(**CONFIG).cpu()
    weights_restored.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))

    # The submodule's keys are local: "0.weight", not "encoder.0.weight".
    fresh_encoder = TinyClassifier(**CONFIG).encoder.cpu()
    encoder_state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    fresh_encoder.load_state_dict(encoder_state)
    with torch.inference_mode():
        encoder_delta = float((model.encoder(inputs) - fresh_encoder(inputs)).abs().max())

    # Same optimizer class/config + its state make the next update reproducible.
    resumed_optimizer = torch.optim.SGD(restored.parameters(), lr=0.05, momentum=0.9)
    resumed_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    train_step(model, optimizer, inputs, labels)
    train_step(restored, resumed_optimizer, inputs, labels)
    next_step_delta = max(float((value - restored.state_dict()[key]).abs().max())
                          for key, value in model.state_dict().items())

    # Separate branch: demonstrate zeros without claiming smaller dense shapes.
    pruned = deepcopy(weights_restored)
    layer = pruned.encoder[0]
    shape_before = list(layer.weight.shape)
    prune.l1_unstructured(layer, name="weight", amount=0.5)
    mask_state_keys = list(pruned.state_dict())
    effective_weight = layer.weight.detach().clone()
    masked_logits = output_logits(pruned, inputs)
    prune.remove(layer, "weight")
    folded_logits = output_logits(pruned, inputs)
    folded_path = output / "pruned-folded.pt"
    torch.save(pruned.state_dict(), folded_path)

    sizes = {}
    for name, path, state in [("weights", weights_path, snapshot),
                              ("checkpoint", checkpoint_path, checkpoint),
                              ("encoder", encoder_path, encoder_state),
                              ("pruned_folded", folded_path, pruned.state_dict())]:
        payload = tensor_payload_bytes(state)
        file_bytes = path.stat().st_size
        sizes[name] = {"tensor_payload_bytes": payload, "actual_file_bytes": file_bytes,
                       "file_minus_tensor_payload_bytes": file_bytes - payload}

    report = {
        "evidence": "seeded synthetic CPU mechanism check; no quality or speed benchmark",
        "torch_version": str(torch.__version__), "seed": SEED, "device": "cpu",
        "config": CONFIG,
        "alias_followed_update": alias_followed_update,
        "deepcopy_snapshot_unchanged": snapshot_isolated,
        "restored_max_logit_difference": float((expected_logits - restored_logits).abs().max()),
        "weights_only_file_max_logit_difference": float((expected_logits - output_logits(weights_restored, inputs)).abs().max()),
        "encoder_max_feature_difference": encoder_delta,
        "resumed_next_step_max_parameter_difference": next_step_delta,
        "weights_only_load_kept_checkpoint_metadata": loaded_checkpoint["completed_steps"] == 1,
        "pruning": {"shape_before": shape_before, "shape_after": list(layer.weight.shape),
                    "zero_count": int((layer.weight == 0).sum()), "elements": layer.weight.numel(),
                    "keys_with_mask": mask_state_keys, "keys_after_remove": list(pruned.state_dict()),
                    "effective_weights_preserved": torch.equal(effective_weight, layer.weight),
                    "remove_max_logit_difference": float((masked_logits - folded_logits).abs().max())},
        "sizes": sizes,
        "size_note": "Dense tensor payload excludes serialization metadata and archive overhead; file bytes are measured, not RAM or inference latency.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-lifecycle"))
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, indent=2))
    print(f"Artifacts written to {args.output}")


if __name__ == "__main__":
    main()
