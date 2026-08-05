"""Validated image-folder indexing with deterministic fixture smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


@dataclass(frozen=True)
class InvalidImage:
    path: Path
    reason: str


class ValidatedImageFolder(Dataset):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root: Path, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.class_names = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.samples: list[tuple[Path, int]] = []
        self.invalid: list[InvalidImage] = []
        for class_name in self.class_names:
            for path in sorted((self.root / class_name).iterdir()):
                if path.suffix.lower() not in self.extensions:
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as error:  # PIL exposes several decoder-specific exceptions.
                    self.invalid.append(InvalidImage(path, type(error).__name__))
                    continue
                self.samples.append((path, self.class_to_index[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            sample = image.convert("RGB")
        if self.transform:
            sample = self.transform(sample)
        return sample, label


def smoke_test() -> dict[str, int]:
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((16, 16))])
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "bee").mkdir()
        (root / "rose").mkdir()
        Image.new("RGB", (12, 10), "gold").save(root / "bee" / "valid.png")
        Image.new("RGB", (9, 14), "crimson").save(root / "rose" / "valid.png")
        (root / "bee" / "broken.jpg").write_bytes(b"not an image")
        dataset = ValidatedImageFolder(root, transform=transform)
        assert dataset.class_names == ["bee", "rose"]
        assert len(dataset) == 2
        assert len(dataset.invalid) == 1
        sample, label = dataset[0]
        assert sample.shape == (3, 16, 16)
        assert label in (0, 1)
        return {"valid_samples": len(dataset), "invalid_samples": len(dataset.invalid)}


if __name__ == "__main__":
    print(smoke_test())

