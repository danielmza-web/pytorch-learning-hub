"""Offline checks for selected Course 2 workflows; no retained assets are modified."""
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import training_comparison
import vision_head
import text_bags
from recall_patterns import collate_bags


def main():
    torch.set_num_threads(2)
    # Accuracy and macro F1 must differ when one class is missed.
    logits = torch.tensor([[4., 0., 0.], [4., 0., 0.], [4., 0., 0.], [0., 0., 4.]])
    measured = training_comparison.metrics(nn.Identity(), DataLoader(
        TensorDataset(logits, torch.tensor([0, 0, 1, 2])), batch_size=3))
    assert measured["confusion_matrix"] == [[2, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert abs(measured["accuracy"] - 0.75) < 1e-6
    assert abs(measured["macro_f1"] - 0.6) < 1e-6
    with TemporaryDirectory(prefix="pytorch-selected-") as directory:
        output = Path(directory)
        comparison = training_comparison.run(output / "training")
        assert len(comparison["trials"]) == 2
        for trial in comparison["trials"]:
            assert sum(map(sum, trial["history"][-1]["confusion_matrix"])) == 48
        assert comparison["trials"][1]["history"][-1]["loss"] < comparison["trials"][1]["history"][0]["loss"]

        # Compare all frozen parameters and buffers, including BatchNorm, to initialization.
        vision_head.run(output / "vision", epochs=1)
        saved = torch.load(output / "vision" / "head-model.pt", weights_only=True)
        torch.manual_seed(31)
        original = resnet18(weights=None)
        original.fc = nn.Linear(original.fc.in_features, 3)
        for name, value in original.state_dict().items():
            if not name.startswith("fc."):
                assert torch.equal(value, saved["state_dict"][name]), name
        assert not torch.equal(original.fc.weight, saved["state_dict"]["fc.weight"])
        assert saved["metadata"]["backbone_weights"] == "random"

        # Exercise the real-file branch offline; class mismatch is rejected.
        image_root = output / "images"
        for split in ("train", "val"):
            for label, colour in (("dark", 20), ("light", 230)):
                folder = image_root / split / label
                folder.mkdir(parents=True)
                Image.new("RGB", (40, 48), (colour,) * 3).save(folder / "sample.png")
        real = vision_head.run(output / "files", data=image_root, epochs=1)
        assert real["classes"] == ["dark", "light"]
        assert real["preprocessing"]["size"] == 224
        (image_root / "val" / "other").mkdir()
        Image.new("RGB", (40, 40)).save(image_root / "val" / "other" / "extra.png")
        try:
            vision_head.run(output / "mismatch", data=image_root, epochs=1)
        except ValueError as error:
            assert "same class" in str(error)
        else:
            raise AssertionError("Class mapping mismatch was accepted")

        report = text_bags.run(output / "text", epochs=3)
        assert report["class_weights"][0] > report["class_weights"][1]
        vocab = report["vocabulary"]
        assert text_bags.encode("unseenword", vocab).tolist() == [1]
        assert text_bags.encode("", vocab).tolist() == [1]
        # Mean pooling is order invariant and ignores an explicitly inserted pad.
        model = text_bags.BagClassifier(len(vocab)).eval()
        ids, offsets, _ = collate_bags([
            (torch.tensor([2, 3]), 0), (torch.tensor([3, 2, 0]), 1)])
        with torch.no_grad():
            logits = model(ids, offsets)
        torch.testing.assert_close(logits[0], logits[1])
        saved_text = torch.load(output / "text" / "text-model.pt", weights_only=True)
        restored = text_bags.BagClassifier(len(saved_text["metadata"]["vocabulary"]))
        restored.load_state_dict(saved_text["state_dict"])
    print("Selected workflows passed: metrics, learning, frozen backbone/buffers, file data, saved contracts and pooling")


if __name__ == "__main__":
    main()
