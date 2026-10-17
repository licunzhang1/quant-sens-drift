"""CLDNN -- the classic backbone most used for signal modulation recognition
(Convolutional + LSTM + DNN).

Reference: Sainath et al., *Convolutional, Long Short-Term Memory, Fully
Connected Deep Neural Networks* (ICASSP 2015); widely adopted for AMC (for
example the RadioML line of work by O'Shea et al.).

Input ``(N, 2, L)``: channel 0=I, 1=Q. Output ``(N, n_classes)``.

A measured constraint on depth
------------------------------
The core quantity rho of this project is the rank correlation of the **layer-wise**
sensitivity ranking, and with too few layers it has almost no statistical power:
at n=5 the smallest possible p-value of a permutation test is about 0.008, and
the confidence interval on rho widens to [-1, 1] (measured).

The first CLDNN implementation (one convolution per stage) had only **5
quantizable weight layers** (3 conv + 2 fc), not enough to support a rank
analysis. It was therefore changed here to **two convolutions per stage**
(``convs_per_stage=2``), which gives 8 conv + 2 fc = **10 weight layers** under
the default configuration.

For comparison: ``ResNet1D`` has 16 weight layers. Both backbones have >=10
layers, which meets the sample-size requirement of the rank analysis.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

__all__ = ["CLDNN"]


class CLDNN(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        in_channels: int = 2,
        conv_channels: Sequence[int] = (64, 64, 128, 128),
        convs_per_stage: int = 2,
        kernel_size: int = 8,
        dropout: float = 0.3,
        lstm_hidden: int = 64,
        fc_hidden: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.convs_per_stage = int(convs_per_stage)

        convs: List[nn.Module] = []
        c_in = in_channels
        for c_out in conv_channels:
            for _ in range(self.convs_per_stage):
                convs += [
                    nn.Conv1d(c_in, c_out, kernel_size,
                              padding=kernel_size // 2, bias=False),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(inplace=True),
                ]
                c_in = c_out
            convs.append(nn.MaxPool1d(2))
        self.conv = nn.Sequential(*convs)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=c_in, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True, bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # tolerate flattened input (N, 2*L)
            x = x.view(x.shape[0], self.in_channels, -1)
        h = self.conv(x)                 # (N, C, L')
        h = h.transpose(1, 2)            # (N, L', C)
        h = self.dropout(h)
        h, _ = self.lstm(h)
        h = h[:, -1, :]                  # take the last time step
        return self.fc(h)

    def weight_layer_names(self) -> List[str]:
        return [n for n, m in self.named_modules()
                if isinstance(m, (nn.Conv1d, nn.Linear))]
