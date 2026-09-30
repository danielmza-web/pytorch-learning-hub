"""Original small patterns for the recall guides; no downloads or course helpers."""
import torch


# --8<-- [start:salt-pepper]
class ImpulseNoise:
    """Replace selected spatial pixels in a float [C,H,W] image in [0,1]."""
    def __init__(self, amount=0.02, salt_fraction=0.5):
        if not 0 <= amount <= 1 or not 0 <= salt_fraction <= 1:
            raise ValueError("Noise probabilities must be in [0, 1]")
        self.amount = amount
        self.salt_fraction = salt_fraction

    def __call__(self, image):
        # One mask shared across channels: a selected RGB pixel is white or black.
        draw = torch.rand_like(image[:1])
        salt = draw < self.amount * self.salt_fraction
        pepper = (draw >= self.amount * self.salt_fraction) & (draw < self.amount)
        noisy = torch.where(salt, torch.ones_like(image), image)
        return torch.where(pepper, torch.zeros_like(image), noisy)
# --8<-- [end:salt-pepper]


# --8<-- [start:masked-mean]
def masked_mean(embeddings, token_ids, pad_id=0):
    # embeddings: [N,L,E]; token_ids: [N,L]
    mask = (token_ids != pad_id).unsqueeze(-1)  # [N,L,1]
    total = (embeddings * mask).sum(dim=1)      # [N,E]
    count = mask.sum(dim=1).clamp_min(1)        # [N,1]
    return total / count                      # empty sequences become zeros
# --8<-- [end:masked-mean]


# --8<-- [start:bag-collate]
def collate_bags(samples):
    # Each sample is (1D token-id tensor, integer label); keep batching on CPU.
    sequences, labels = zip(*samples)
    sequences = [s if s.numel() else torch.tensor([1]) for s in sequences]
    lengths = torch.tensor([s.numel() for s in sequences], dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]])
    return torch.cat(sequences), offsets, torch.tensor(labels, dtype=torch.long)
# --8<-- [end:bag-collate]


# --8<-- [start:accumulation]
def train_accumulated(model, loader, optimizer, device, microbatches=4):
    """Single device; unweighted classification; average by actual group size."""
    if microbatches < 1:
        raise ValueError("microbatches must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    group_samples = 0
    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        loss_fn(model(x), y).backward()  # add gradients; no update yet
        group_samples += y.numel()
        if step % microbatches == 0 or step == len(loader):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_samples)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            group_samples = 0
# --8<-- [end:accumulation]
