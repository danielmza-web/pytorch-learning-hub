"""Offline integration checks for real split/checkpoint paths and retained evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import copy
import json
import sys
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import emnist_model as emnist
import nature_cnn as nature
import training_comparison as comparison
from validation_patterns import BestState, split_indices


class FakeImages(Dataset):
    def __init__(self, root=None, train=True, download=False, transform=None, target_transform=None, nature_mode=False):
        self.train, self.transform, self.target_transform = train, transform, target_transform
        self.classes = nature.CLASS_NAMES if nature_mode else emnist.LETTERS
        self.targets = list(range(len(self.classes))) * (4 if train else 1)
        self.mode = nature_mode

    def __len__(self): return len(self.targets)

    def __getitem__(self, index):
        label = self.targets[index]
        image = Image.new("RGB" if self.mode else "L", (32, 32) if self.mode else (28, 28), color=20 + label * 5)
        if self.transform: image = self.transform(image)
        if not self.mode: label += 1
        if self.target_transform: label = self.target_transform(label)
        return image, label


def source(dataset):
    while hasattr(dataset, "dataset"): dataset = dataset.dataset
    return dataset


def check_split_and_restore():
    x = torch.tensor([[1., 2.], [3., 4.]])
    w = torch.tensor([[-4., 3.], [2., -1.]], requires_grad=True)
    logits = x @ w.T
    torch.testing.assert_close(logits, torch.tensor([[2., 0.], [0., 2.]]))
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1])); loss.backward()
    assert abs(loss.item() - .126928) < 1e-6
    torch.testing.assert_close(w.grad, torch.tensor([[.1192029, .1192029], [-.1192029, -.1192029]]))
    train, val = split_indices(133, .2, 17)
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(133))
    assert (train, val) == split_indices(133, .2, 17)
    for n, fraction in [(1, .2), (10, 0), (10, 1)]:
        try: split_indices(n, fraction)
        except ValueError: pass
        else: raise AssertionError("Invalid split accepted")
    model = nn.Linear(2, 1)
    expected = copy.deepcopy(model.state_dict())
    best = BestState(); best.consider(model, 1., 1)
    with torch.no_grad(): model.weight.add_(10)
    best.consider(model, 2., 2); best.restore(model)
    assert best.epoch == 1
    for k in expected: torch.testing.assert_close(model.state_dict()[k], expected[k])


def check_actual_runs(output):
    for module in [emnist, nature]:
        validation_states, test_states = [], []
        function_name = "evaluate" if module is emnist else "epoch"
        original = getattr(module, function_name)
        def measured(model, loader, *args):
            result = original(model, loader, *args)
            is_training = module is nature and args[-1] is True
            if is_training: return result
            state = copy.deepcopy(model.state_dict())
            if source(loader.dataset).train:
                validation_states.append(state)
                # Force the second validation check to be worse: final epoch must not be kept.
                return (float(len(validation_states)), *result[1:])
            test_states.append(state)
            return result
        folder = output / ("emnist" if module is emnist else "nature")
        args = SimpleNamespace(seed=7, device="cpu", data_dir=output / "unused", output_dir=folder,
            full=False, epochs=2, full_epochs=2, train_limit=52, test_limit=26,
            train_per_class=4, test_per_class=1, batch_size=16, learning_rate=.001,
            weight_decay=.0001, validation_fraction=.2)
        dataset_patch = patch.object(emnist, "load_emnist", side_effect=lambda root, **kwargs: FakeImages(root, **kwargs)) if module is emnist else patch.object(nature.datasets, "CIFAR100", side_effect=lambda root, **kwargs: FakeImages(root, nature_mode=True, **kwargs))
        figure_patch = patch.object(module, "save_figures" if module is emnist else "save_artifacts", side_effect=lambda *args: folder.mkdir(parents=True, exist_ok=True))
        with dataset_patch, figure_patch, patch.object(module, function_name, side_effect=measured):
            report = module.run(args)
        assert len(validation_states) == 2 and len(test_states) == 1
        assert report["best_epoch"] == 1
        splits = report["split"]
        assert not set(splits["training_source_indices"]) & set(splits["validation_source_indices"])
        checkpoint = torch.load(folder / ("emnist-model.pt" if module is emnist else "nature-cnn.pt"), weights_only=True)
        for key, expected in validation_states[0].items():
            torch.testing.assert_close(test_states[0][key], expected)
            torch.testing.assert_close(checkpoint["model_state"][key], expected)
        assert checkpoint["best_epoch"] == 1 and checkpoint["normalization"]


def check_benchmark_and_evidence():
    torch.manual_seed(17)
    x = torch.randn(133, 6); y = torch.arange(133) % 3
    model = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 3))
    report = comparison.benchmark(model, TensorDataset(x, y), torch.device("cpu"), repetitions=1)
    assert all(v["updates"] == 3 and v["samples"] == 133 for v in report["variants"])
    assert report["variants"][1]["max_parameter_difference_from_physical_fp32"] < 1e-6
    assert report["amp"].startswith("skipped")
    empty = comparison.metrics(nn.Identity(), DataLoader(TensorDataset(torch.empty(0, 3), torch.empty(0, dtype=torch.long))))
    assert empty["accuracy"] is None and empty["loss"] is None
    assert empty["macro_f1"] is None and empty["recall_by_class"] == [None] * 3
    directory = ROOT / "docs/assets/data/recall-2026-10-01"
    training = json.loads((directory / "comparison.json").read_text())
    for trial in training["trials"]:
        for epoch in trial["history"]:
            matrix = epoch["confusion_matrix"]
            assert sum(map(sum, matrix)) == training["validation_samples"]
            expected_accuracy = sum(matrix[i][i] for i in range(3)) / 48
            assert abs(expected_accuracy - epoch["accuracy"]) < 1e-6
    text = json.loads((directory / "predictions.json").read_text())
    assert text["unknown_input"]["ids"] == [1]
    assert text["order_check"]["max_logit_difference"] == 0
    vision = json.loads((directory / "report.json").read_text())
    assert vision["frozen_parameters_unchanged"] and vision["frozen_buffers_unchanged"] and vision["head_changed"]
    for name in ["recall-visuals.js", "recall-visuals.css"]:
        folder = "javascripts" if name.endswith("js") else "stylesheets"
        parent_copy = ROOT.parent / "site/learn/pytorch/assets" / name
        if parent_copy.exists(): assert parent_copy.read_bytes() == (ROOT / "docs/assets" / folder / name).read_bytes()


def main():
    torch.set_num_threads(2)
    check_split_and_restore()
    with TemporaryDirectory(prefix="pytorch-contracts-") as directory: check_actual_runs(Path(directory))
    check_benchmark_and_evidence()
    print("Learning contracts passed: disjoint splits, best-state restoration before final test, incomplete accumulation and retained reports")


if __name__ == "__main__": main()
