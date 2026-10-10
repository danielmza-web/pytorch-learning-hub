"""Offline lifecycle invariants using temporary output, no maintained assets."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
from torch.nn.utils import prune

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from model_lifecycle import (CONFIG, SEED, TinyClassifier, output_logits, run,
                             synthetic_batch, tensor_payload_bytes, train_step)


class LifecycleTests(unittest.TestCase):
    def test_snapshot_survives_later_parameter_update(self):
        torch.manual_seed(12)
        model = TinyClassifier(**CONFIG)
        live = model.state_dict()
        snapshot = deepcopy(live)
        before = {key: value.clone() for key, value in snapshot.items()}
        for key in snapshot:
            self.assertNotEqual(snapshot[key].data_ptr(), live[key].data_ptr())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(2)
        self.assertTrue(torch.equal(live["head.weight"], model.head.weight))
        self.assertFalse(torch.equal(live["head.weight"], snapshot["head.weight"]))
        for key in snapshot:
            torch.testing.assert_close(snapshot[key], before[key], rtol=0, atol=0)

    def test_roundtrip_parts_metadata_and_optimizer_resume(self):
        temporary_parent = Path(__file__).resolve().parents[1] / "artifacts"
        temporary_parent.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix="lifecycle-test-", dir=temporary_parent) as directory:
            folder = Path(directory)
            report = run(folder)
            saved_report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_report, report)
            checkpoint = torch.load(folder / "checkpoint.pt", weights_only=True, map_location="cpu")
            # This flag leaves basic metadata and optimizer state in the result.
            self.assertEqual(checkpoint["completed_steps"], 1)
            self.assertEqual(checkpoint["config"], CONFIG)
            self.assertEqual(checkpoint["labels"], ["class-zero", "class-one"])
            self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
            weights = torch.load(folder / "weights.pt", weights_only=True, map_location="cpu")
            self.assertEqual(set(weights), set(TinyClassifier(**CONFIG).state_dict()))
            self.assertTrue(all(value.device.type == "cpu" for value in weights.values()))
            restored = TinyClassifier(**checkpoint["config"])
            restored.load_state_dict(weights)
            from_checkpoint = TinyClassifier(**checkpoint["config"])
            from_checkpoint.load_state_dict(checkpoint["model_state_dict"])
            inputs, labels = synthetic_batch()
            torch.manual_seed(SEED)
            reference = TinyClassifier(**CONFIG)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05, momentum=0.9)
            train_step(reference, reference_optimizer, inputs, labels)
            torch.testing.assert_close(output_logits(restored, inputs), output_logits(reference, inputs), rtol=0, atol=0)
            torch.testing.assert_close(output_logits(restored, inputs), output_logits(from_checkpoint, inputs), rtol=0, atol=0)
            encoder = TinyClassifier(**CONFIG).encoder
            parts = torch.load(folder / "encoder.pt", weights_only=True, map_location="cpu")
            self.assertEqual(set(parts), {"0.weight", "0.bias"})
            encoder.load_state_dict(parts)
            torch.testing.assert_close(encoder(inputs), restored.encoder(inputs), rtol=0, atol=0)
            optimizers = [torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
                          for model in (restored, from_checkpoint)]
            for optimizer in optimizers:
                optimizer.load_state_dict(deepcopy(checkpoint["optimizer_state_dict"]))
            for model, optimizer in zip((restored, from_checkpoint), optimizers):
                train_step(model, optimizer, inputs, labels)
            train_step(reference, reference_optimizer, inputs, labels)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, from_checkpoint.state_dict()[key], rtol=0, atol=0)
                torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
            for key in ("restored_max_logit_difference", "weights_only_file_max_logit_difference",
                        "encoder_max_feature_difference", "resumed_next_step_max_parameter_difference"):
                self.assertEqual(report[key], 0.0)
            self.assertTrue(report["deepcopy_snapshot_unchanged"])
            self.assertTrue(report["alias_followed_update"])
            self.assertEqual(report["sizes"]["weights"]["tensor_payload_bytes"], 232)
            self.assertGreater(report["sizes"]["checkpoint"]["tensor_payload_bytes"], 232)
            self.assertEqual(report["sizes"]["pruned_folded"]["tensor_payload_bytes"], 232)
            folded_state = torch.load(folder / "pruned-folded.pt", weights_only=True, map_location="cpu")
            self.assertEqual(set(folded_state), set(weights))
            self.assertEqual(tuple(folded_state["encoder.0.weight"].shape), (8, 4))
            self.assertEqual(int((folded_state["encoder.0.weight"] == 0).sum()), 16)
            independent_pruned = TinyClassifier(**CONFIG)
            independent_pruned.load_state_dict(weights)
            prune.l1_unstructured(independent_pruned.encoder[0], "weight", amount=0.5)
            prune.remove(independent_pruned.encoder[0], "weight")
            from_folded_file = TinyClassifier(**CONFIG)
            from_folded_file.load_state_dict(folded_state)
            torch.testing.assert_close(output_logits(from_folded_file, inputs),
                                       output_logits(independent_pruned, inputs), rtol=0, atol=0)
            for name, filename in (("weights", "weights.pt"), ("checkpoint", "checkpoint.pt"),
                                   ("encoder", "encoder.pt"), ("pruned_folded", "pruned-folded.pt")):
                size = report["sizes"][name]
                self.assertEqual(size["actual_file_bytes"], (folder / filename).stat().st_size)
                self.assertEqual(size["actual_file_bytes"] - size["tensor_payload_bytes"],
                                 size["file_minus_tensor_payload_bytes"])

    def test_pruning_keeps_dense_shape_and_remove_folds_mask(self):
        torch.manual_seed(29)
        model = TinyClassifier(**CONFIG)
        layer = model.encoder[0]
        shape, payload = layer.weight.shape, tensor_payload_bytes(model.state_dict())
        prune.l1_unstructured(layer, "weight", amount=0.5)
        self.assertIn("encoder.0.weight_orig", model.state_dict())
        self.assertIn("encoder.0.weight_mask", model.state_dict())
        self.assertEqual(layer.weight.shape, shape)
        self.assertEqual(int((layer.weight_mask == 0).sum()), 16)
        self.assertEqual(int((layer.weight == 0).sum()), 16)
        effective = layer.weight.detach().clone()
        inputs, _ = synthetic_batch()
        masked_logits = output_logits(model, inputs)
        prune.remove(layer, "weight")
        self.assertNotIn("encoder.0.weight_orig", model.state_dict())
        self.assertNotIn("encoder.0.weight_mask", model.state_dict())
        self.assertIn("encoder.0.weight", model.state_dict())
        self.assertEqual(layer.weight.shape, shape)
        self.assertEqual(tensor_payload_bytes(model.state_dict()), payload)
        torch.testing.assert_close(layer.weight, effective, rtol=0, atol=0)
        torch.testing.assert_close(output_logits(model, inputs), masked_logits, rtol=0, atol=0)
        restored = TinyClassifier(**CONFIG)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(output_logits(restored, inputs), masked_logits, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
