"""Original tiny text classifier: vocabulary, offsets, class weights and saved contract."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from recall_patterns import collate_bags

TRAIN = [
    ("clear image", 1), ("sharp picture", 1), ("good bright photo", 1),
    ("clear sharp photo", 1), ("good image", 1), ("bright picture", 1),
    ("blurry dark image", 0), ("bad blurry photo", 0),
]
VALIDATION = [("good sharp image", 1), ("bad dark picture", 0),
              ("clear bright photo", 1), ("blurry image", 0)]


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def vocabulary(samples):
    words = sorted({word for text, _ in samples for word in tokenize(text)})
    return {"<pad>": 0, "<unk>": 1, **{word: i + 2 for i, word in enumerate(words)}}


def encode(text, vocab):
    ids = [vocab.get(word, 1) for word in tokenize(text)]
    return torch.tensor(ids or [1], dtype=torch.long)


class BagClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, 12, mode="mean", padding_idx=0)
        self.head = nn.Linear(12, 2)

    def forward(self, ids, offsets):
        return self.head(self.embedding(ids, offsets))


def run(output: Path, epochs=20):
    torch.set_num_threads(2)
    torch.manual_seed(53)
    vocab = vocabulary(TRAIN)  # no validation words are used to build the vocabulary
    encoded = [(encode(text, vocab), label) for text, label in TRAIN]
    loader = DataLoader(encoded, batch_size=3, shuffle=True, collate_fn=collate_bags,
                        generator=torch.Generator().manual_seed(59))
    counts = torch.bincount(torch.tensor([label for _, label in TRAIN]), minlength=2)
    weights = counts.sum() / (2 * counts.float())  # [class 0, class 1]
    model = BagClassifier(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    history = []
    for epoch in range(epochs):
        model.train()
        for ids, offsets, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, offsets)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            ids, offsets, labels = collate_bags(
                [(encode(text, vocab), label) for text, label in VALIDATION])
            predicted = model(ids, offsets).argmax(1)
            accuracy = (predicted == labels).float().mean().item()
        history.append({"epoch": epoch + 1, "validation_accuracy": accuracy})
    predictions = [{"text": text, "target": label, "predicted": int(pred)}
                   for (text, label), pred in zip(VALIDATION, predicted)]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "data": "tiny original toy phrases; not a sentiment benchmark",
        "vocabulary": vocab, "labels": ["poor-image-description", "good-image-description"],
        "tokenization": "lowercase ASCII word regex", "empty_input": "<unk>",
        "pooling": "mean; no word order; no padding required",
        "class_weights": weights.tolist(), "history": history, "predictions": predictions,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": report}, output / "text-model.pt")
    (output / "predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(predictions, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/text-bags"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    run(args.output, args.epochs)
