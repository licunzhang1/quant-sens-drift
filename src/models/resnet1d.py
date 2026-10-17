"""1D-ResNet -- the validation backbone.

Its purpose is to answer a question every reviewer will ask:
**"does your conclusion hold only for CLDNN?"**
Re-running with a structurally different backbone, and obtaining the same
conclusion, makes it more likely to be **architecture-independent**.

Input ``(N, 2, L)``, output ``(N, n_classes)``.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResNet1D", "BasicBlock1D"]


class BasicBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return F.relu(y + x, inplace=True)


class ResNet1D(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        in_channels: int = 2,
        base_channels: int = 32,
        n_blocks: Sequence[int] = (2, 2, 2),
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        stages: List[nn.Module] = []
        c = base_channels
        for i, nb in enumerate(n_blocks):
            if i > 0:
                c_out = c * 2
                stages.append(nn.Sequential(
                    nn.Conv1d(c, c_out, kernel_size=1, bias=False),
                    nn.BatchNorm1d(c_out),
                    nn.MaxPool1d(2),
                ))
                c = c_out
            for _ in range(nb):
                stages.append(BasicBlock1D(c, kernel_size, dropout))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.shape[0], self.in_channels, -1)
        return self.head(self.stages(self.stem(x)))

    def weight_layer_names(self) -> List[str]:
        return [n for n, m in self.named_modules()
                if isinstance(m, (nn.Conv1d, nn.Linear))]
