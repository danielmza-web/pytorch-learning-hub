"""Seeded split and best-state selection shared by the image demonstrations."""
import copy
import torch


def split_indices(total, fraction=0.2, seed=42):
    if total < 2 or not 0 < fraction < 1:
        raise ValueError("Need at least two samples and a validation fraction between 0 and 1")
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
    count = min(total - 1, max(1, round(total * fraction)))
    return order[count:], order[:count]


class BestState:
    """Retain an independent snapshot, minimizing validation loss."""
    def __init__(self):
        self.loss = float("inf")
        self.epoch = None
        self.state = None

    def consider(self, model, loss, epoch):
        if loss < self.loss:
            self.loss, self.epoch = float(loss), epoch
            self.state = copy.deepcopy(model.state_dict())

    def restore(self, model):
        if self.state is None:
            raise ValueError("No finite validation result was retained")
        model.load_state_dict(self.state)
