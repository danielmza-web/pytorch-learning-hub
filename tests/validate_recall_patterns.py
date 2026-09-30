"""Check educational invariants with tiny synthetic tensors, no downloads."""
from pathlib import Path
import sys
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from recall_patterns import ImpulseNoise, masked_mean, collate_bags, train_accumulated


def main():
    torch.manual_seed(17)
    image = torch.full((3, 7, 9), 0.4)
    assert torch.equal(ImpulseNoise(0)(image), image)
    assert torch.equal(ImpulseNoise(1, 1)(image), torch.ones_like(image))
    assert torch.equal(ImpulseNoise(1, 0)(image), torch.zeros_like(image))
    assert torch.equal(image, torch.full_like(image, 0.4))  # preserve source storage
    noise = ImpulseNoise(0.5)(image)
    assert torch.equal(noise[0], noise[1]) and torch.equal(noise[1], noise[2])
    assert noise.min() >= 0 and noise.max() <= 1

    embeddings = torch.tensor([[[2., 4.], [4., 8.], [99., 99.]], [[9., 9.], [9., 9.], [9., 9.]]])
    ids = torch.tensor([[2, 3, 0], [0, 0, 0]])
    torch.testing.assert_close(masked_mean(embeddings, ids), torch.tensor([[3., 6.], [0., 0.]]))

    sequences = [torch.tensor([2, 4, 6]), torch.tensor([3, 5]), torch.tensor([], dtype=torch.long)]
    flat, offsets, labels = collate_bags(list(zip(sequences, [0, 1, 2])))
    assert offsets.tolist() == [0, 3, 5] and labels.tolist() == [0, 1, 2]
    bag = nn.EmbeddingBag(10, 4, mode="mean")
    actual = bag(flat, offsets)
    expected = torch.stack([bag.weight[s].mean(0) for s in [sequences[0], sequences[1], torch.tensor([1])]])
    torch.testing.assert_close(actual, expected)

    # Same sample-weighted gradients, including an unequal final microbatch.
    x = torch.randn(11, 5)
    y = torch.tensor([0, 1, 2, 0, 2, 1, 0, 2, 1, 1, 0])
    base = nn.Linear(5, 3)
    accumulated = copy.deepcopy(base)
    large_batch = copy.deepcopy(base)
    small_loader = DataLoader(TensorDataset(x, y), batch_size=3)
    big_loader = DataLoader(TensorDataset(x, y), batch_size=6)
    small_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.03)
    big_optimizer = torch.optim.SGD(large_batch.parameters(), lr=0.03)
    train_accumulated(accumulated, small_loader, small_optimizer, torch.device("cpu"), 2)
    for inputs, targets in big_loader:
        big_optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(large_batch(inputs), targets).backward()
        big_optimizer.step()
    for actual, expected in zip(accumulated.parameters(), large_batch.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    print("Recall patterns verified: noise, masked pooling, offsets and final-group accumulation")


if __name__ == "__main__":
    main()
