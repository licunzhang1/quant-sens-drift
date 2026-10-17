"""torch Dataset wrappers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ["IQDataset", "make_loader"]


class IQDataset(Dataset):
    """Wrap an ``(N, 2, L)`` numpy IQ array as a torch Dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 transform: Optional[object] = None):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"x should be (N, 2, L), got {x.shape}")
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.transform = transform
        assert len(self.x) == len(self.y), "x and y have different lengths"

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xi = self.x[i]
        if self.transform is not None:
            xi = self.transform(xi)
        return xi, self.y[i]


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> DataLoader:
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
    return DataLoader(
        IQDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        generator=gen,
    )
