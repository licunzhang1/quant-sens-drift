"""经典 AMC 特征：高阶累积量 + 包络统计。

为什么项目里需要它
------------------
两个用途，一次实现：

1. **参考模型的输入表示**（``tests/smoke_pipeline.py``）。
   原始 IQ 波形里，调制方式的信息藏在**分布**里而不是某几个采样点上，
   普通 MLP 很难自己学出来（实测：训练 1.0 / 测试 0.50，只有 BPSK 能分）。
   换成累积量特征后模型能真正泛化，"域间隙"才有意义。

2. **经典 DSP 基线臂**（研究设计里的切口 2）。
   回答"深度学习到底比传统方法强多少"——这个问题在文献里被大量宣称，
   但很少有人用**同样严格的划分协议**去比。

累积量定义（复数随机变量 z，零均值）
-----------------------------------
C20 = E[z^2]
C21 = E[|z|^2]
C40 = E[z^4] - 3·E[z^2]^2
C41 = E[z^3z*] - 3·E[z^2]·E[|z|^2]
C42 = E[|z|^4] - |E[z^2]|^2 - 2·E[|z|^2]^2
C60 = E[z^6] - 15·E[z^2]E[z^4] + 30·E[z^2]^3
C63 = E[|z|^6] - 9·E[|z|^4]E[|z|^2] - 6·E[z^2*]... （见实现，采用常用的近似形式）

这些量对星座形状敏感，且天然对旋转/尺度有一定不变性，是经典 AMC 的标准特征。
"""

from __future__ import annotations

from typing import List

import numpy as np

__all__ = ["FEATURE_NAMES", "amc_features", "feature_dim"]

_EPS = 1e-12

FEATURE_NAMES: List[str] = [
    "abs_C20",
    "abs_C40",
    "abs_C41",
    "abs_C42",
    "abs_C60",
    "abs_C63",
    "mag_kurtosis",
    "mag_var",
    "mag_skew",
    "phase_hist_entropy",
    "abs_re_kurtosis",
    "abs_im_kurtosis",
    "mag_p90_p10",
    "mag_iqr",
]


def feature_dim() -> int:
    return len(FEATURE_NAMES)


def _phase_entropy(z: np.ndarray, n_bins: int = 16) -> float:
    """相位直方图熵（归一化到 [0,1]）。BPSK 的相位分布最集中。"""
    ph = np.angle(z)
    hist, _ = np.histogram(ph, bins=n_bins, range=(-np.pi, np.pi))
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    h = -np.sum(p * np.log(p))
    return float(h / np.log(n_bins))


def _kurtosis(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size < 4:
        return 0.0
    m = v.mean()
    s = v.std()
    if s < _EPS:
        return 0.0
    return float(np.mean(((v - m) / s) ** 4) - 3.0)


def _skew(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size < 3:
        return 0.0
    m = v.mean()
    s = v.std()
    if s < _EPS:
        return 0.0
    return float(np.mean(((v - m) / s) ** 3))


def amc_features(x: np.ndarray) -> np.ndarray:
    """把 ``(N, 2, L)`` 的 IQ 批量转成 ``(N, F)`` 特征矩阵。

    ``L`` 方向上的所有采样点都参与统计（脉压相关使它们非独立，
    但经验累积量仍然具有判别力——这是经典 AMC 的常规做法）。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"输入应为 (N, 2, L)，收到 {x.shape}")
    n = x.shape[0]
    z = x[:, 0, :] + 1j * x[:, 1, :]  # (N, L)

    # 每个样本各自归一化到单位功率，保证尺度不变
    p = np.mean(np.abs(z) ** 2, axis=1, keepdims=True)
    z = z / np.sqrt(np.maximum(p, _EPS))

    z2 = z * z
    z3 = z2 * z
    z4 = z2 * z2
    z6 = z3 * z3
    zc = np.conj(z)
    m2 = np.abs(z) ** 2
    m4 = m2 * m2
    m6 = m4 * m2
    m3 = m2 * np.abs(z)

    E = lambda a: np.mean(a, axis=1)  # noqa: E731

    E_z2 = E(z2)
    E_z4 = E(z4)
    E_z6 = E(z6)
    E_m2 = E(m2)
    E_m4 = E(m4)
    E_m6 = E(m6)
    E_z3c = E(z3 * zc)
    E_z5c = E(z3 * z2 * zc)

    C20 = E_z2
    C40 = E_z4 - 3.0 * E_z2 ** 2
    C41 = E_z3c - 3.0 * E_z2 * E_m2
    C42 = E_m4 - np.abs(E_z2) ** 2 - 2.0 * E_m2 ** 2
    C60 = E_z6 - 15.0 * E_z2 * E_z4 + 30.0 * E_z2 ** 3
    C63 = (
        E_m6
        - 9.0 * E_m4 * E_m2
        - 6.0 * np.abs(E_z2) ** 2 * E_m2
        + 18.0 * E_m2 ** 3
    )

    mag = np.abs(z)
    mag_kurt = np.array([_kurtosis(mag[i]) for i in range(n)])
    mag_var = np.var(mag, axis=1)
    mag_skew = np.array([_skew(mag[i]) for i in range(n)])
    ph_ent = np.array([_phase_entropy(z[i]) for i in range(n)])
    abs_re_kurt = np.array([_kurtosis(np.abs(z[i].real)) for i in range(n)])
    abs_im_kurt = np.array([_kurtosis(np.abs(z[i].imag)) for i in range(n)])

    p90 = np.percentile(mag, 90, axis=1)
    p10 = np.percentile(mag, 10, axis=1)
    q75 = np.percentile(mag, 75, axis=1)
    q25 = np.percentile(mag, 25, axis=1)
    mag_p90_p10 = p90 / np.maximum(p10, _EPS)
    mag_iqr = q75 - q25

    feats = np.stack(
        [
            np.abs(C20),
            np.abs(C40),
            np.abs(C41),
            np.abs(C42),
            np.abs(C60),
            np.abs(C63),
            mag_kurt,
            mag_var,
            mag_skew,
            ph_ent,
            abs_re_kurt,
            abs_im_kurt,
            mag_p90_p10,
            mag_iqr,
        ],
        axis=1,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    assert feats.shape == (n, len(FEATURE_NAMES)), feats.shape
    return feats
