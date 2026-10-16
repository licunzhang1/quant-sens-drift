"""CLDNN —— 信号调制识别最常用的经典骨干（Convolutional + LSTM + DNN）。

参考：Sainath et al., *Convolutional, Long Short-Term Memory, Fully Connected
Deep Neural Networks* (ICASSP 2015)；在 AMC 上被广泛采用（如 O'Shea et al. 的
RadioML 系列工作）。

输入 ``(N, 2, L)``：通道 0=I、1=Q。输出 ``(N, n_classes)``。

关于深度的一个实测约束
----------------------
本项目的核心量 ρ 是**逐层**敏感性排序的秩相关，层数太少时几乎没有统计功效：
n=5 时置换检验的最小可能 p 值约 0.008，且 ρ 的置信区间会宽到 [-1, 1]（实测）。

最初的 CLDNN 实现（每个 stage 一个卷积）只有 **5 个可量化权重层**
（3 conv + 2 fc），不足以支撑秩分析。因此这里改成每个 stage **两个卷积**
（``convs_per_stage=2``），默认配置下得到 8 conv + 2 fc = **10 个权重层**。

对照：``ResNet1D`` 有 16 个权重层。两个骨干的层数都 ≥10，符合秩分析的样本量要求。
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
        if x.dim() == 2:  # 兼容展平输入 (N, 2*L)
            x = x.view(x.shape[0], self.in_channels, -1)
        h = self.conv(x)                 # (N, C, L')
        h = h.transpose(1, 2)            # (N, L', C)
        h = self.dropout(h)
        h, _ = self.lstm(h)
        h = h[:, -1, :]                  # 取最后一个时间步
        return self.fc(h)

    def weight_layer_names(self) -> List[str]:
        return [n for n, m in self.named_modules()
                if isinstance(m, (nn.Conv1d, nn.Linear))]
