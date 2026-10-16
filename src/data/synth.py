"""合成 IQ 信号生成 —— 零下载、完全可控的偏移构造。

为什么默认用合成数据
--------------------
本项目要研究的是**分布偏移下的量化敏感性**，因此必须能**精确控制偏移的种类与强度**。
公开数据集（RML2018.01A 等）通常只给一个固定信道条件，无法做受控偏移实验。
合成数据的另一个好处是：零下载、可复现、可无限扩样。

若要接入公开数据集，实现一个新的 ``Dataset`` 即可，本模块不必改动。

输出的样本格式与 RadioML 一致：``X: (N, 2, L)``，通道 0=I、1=Q。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MODULATIONS",
    "make_constellation",
    "rrc_filter",
    "pulse_shape",
    "apply_channel",
    "make_iq_dataset",
    "ChannelConfig",
]


# --------------------------------------------------------------------------
# 星座图
# --------------------------------------------------------------------------
def _pam_levels(m: int) -> np.ndarray:
    """m-PAM 的归一化电平（单位平均能量）。"""
    levels = np.arange(-(m - 1), m, 2, dtype=np.float64)
    return levels / np.sqrt(np.mean(levels ** 2))


def make_constellation(name: str) -> np.ndarray:
    """返回单位平均能量的复星座点。"""
    name = name.lower()
    if name == "bpsk":
        pts = np.array([-1.0, 1.0], dtype=np.complex128)
    elif name == "qpsk":
        pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex128) / np.sqrt(2)
    elif name == "8psk":
        k = np.arange(8)
        pts = np.exp(1j * 2 * np.pi * k / 8)
    elif name == "16qam":
        lv = _pam_levels(4)
        pts = np.array([a + 1j * b for a in lv for b in lv], dtype=np.complex128)
    elif name == "8qam":
        lv = _pam_levels(4)[1::2]  # ±1/sqrt(2) 量级
        outer = np.array([a + 1j * b for a in lv for b in lv], dtype=np.complex128)
        lv2 = _pam_levels(2)
        inner = np.array([a + 1j * b for a in lv2 for b in lv2], dtype=np.complex128)
        pts = np.concatenate([outer, inner])
    else:
        raise ValueError(f"未知调制方式: {name}")
    pts = pts.astype(np.complex128)
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


MODULATIONS: Tuple[str, ...] = ("bpsk", "qpsk", "8psk", "8qam", "16qam")


# --------------------------------------------------------------------------
# 脉冲成形
# --------------------------------------------------------------------------
def rrc_filter(beta: float, sps: int, span: int = 6) -> np.ndarray:
    """根升余弦滤波器（单位能量）。"""
    if sps < 1:
        raise ValueError("sps 必须 >= 1")
    n = span * sps
    t = np.arange(-n / 2, n / 2 + 1, dtype=np.float64) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-8:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                - (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(
                np.pi * ti * (1 + beta)
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    e = np.sum(h ** 2)
    if e < 1e-12:
        raise ValueError("RRC 滤波器能量异常")
    return h / np.sqrt(e)


def pulse_shape(symbols: np.ndarray, sps: int, beta: float, span: int = 6) -> np.ndarray:
    """上采样 + RRC 成形。"""
    up = np.zeros(symbols.size * sps, dtype=np.complex128)
    up[::sps] = symbols
    h = rrc_filter(beta, sps, span)
    return np.convolve(up, h, mode="same")


# --------------------------------------------------------------------------
# 信道
# --------------------------------------------------------------------------
@dataclass
class ChannelConfig:
    """信道损伤配置。Domain A / Domain B 各用一份。"""

    snr_db: Tuple[float, float] = (10.0, 18.0)   # 区间，样本间随机取
    cfo: float = 0.0                             # 归一化载波频偏（相对符号率）
    phase_noise_deg: float = 0.0                 # 相位噪声标准差（度）
    iq_gain_db: float = 0.0                      # IQ 幅度失衡
    iq_phase_deg: float = 0.0                    # IQ 相位失衡（度）
    gain_db: float = 0.0                         # 接收机整体增益偏移

    @classmethod
    def from_dict(cls, d: Optional[Dict] = None) -> "ChannelConfig":
        d = dict(d or {})
        snr = d.pop("snr_db", (10.0, 18.0))
        if np.isscalar(snr):
            snr = (float(snr), float(snr))
        return cls(snr_db=(float(snr[0]), float(snr[1])), **d)

    def describe(self) -> str:
        return (
            f"SNR∈[{self.snr_db[0]:g},{self.snr_db[1]:g}]dB"
            f", CFO={self.cfo:g}, PN={self.phase_noise_deg:g}°"
            f", IQg={self.iq_gain_db:g}dB, IQφ={self.iq_phase_deg:g}°"
            f", gain={self.gain_db:g}dB"
        )


def _add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p_sig = np.mean(np.abs(x) ** 2)
    if p_sig < 1e-20:
        return x
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(p_noise / 2.0)
    n = sigma * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    return x + n


def _add_cfo(x: np.ndarray, cfo: float) -> np.ndarray:
    if cfo == 0.0:
        return x
    n = np.arange(x.size, dtype=np.float64)
    return x * np.exp(1j * 2.0 * np.pi * cfo * n)


def _add_phase_noise(x: np.ndarray, sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    if sigma_deg <= 0:
        return x
    sigma = np.deg2rad(sigma_deg)
    walk = np.cumsum(rng.standard_normal(x.size)) * sigma
    return x * np.exp(1j * walk)


def _add_iq_imbalance(x: np.ndarray, gain_db: float, phase_deg: float) -> np.ndarray:
    if gain_db == 0.0 and phase_deg == 0.0:
        return x
    eps = 10.0 ** (gain_db / 20.0) - 1.0
    phi = np.deg2rad(phase_deg)
    i, q = x.real, x.imag
    i2 = (1.0 + eps) * i
    q2 = (1.0 - eps) * (np.cos(phi) * q + np.sin(phi) * i)
    return i2 + 1j * q2


def apply_channel(
    x: np.ndarray,
    cfg: ChannelConfig,
    rng: np.random.Generator,
    snr_db: Optional[float] = None,
) -> np.ndarray:
    """按配置施加信道损伤。顺序：IQ 失衡 → CFO → 相位噪声 → 增益 → AWGN。"""
    y = np.asarray(x, dtype=np.complex128)
    y = _add_iq_imbalance(y, cfg.iq_gain_db, cfg.iq_phase_deg)
    y = _add_cfo(y, cfg.cfo)
    y = _add_phase_noise(y, cfg.phase_noise_deg, rng)
    if cfg.gain_db != 0.0:
        y = y * (10.0 ** (cfg.gain_db / 20.0))
    snr = float(snr_db) if snr_db is not None else float(
        rng.uniform(cfg.snr_db[0], cfg.snr_db[1])
    )
    y = _add_awgn(y, snr, rng)
    # 统一归一化到单位平均功率（与 RadioML 一致）
    p = np.mean(np.abs(y) ** 2)
    if p > 1e-20:
        y = y / np.sqrt(p)
    return y


# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------
def make_iq_dataset(
    n_samples: int,
    cfg: ChannelConfig,
    *,
    modulations: Sequence[str] = ("bpsk", "qpsk", "8psk", "16qam"),
    n_symbols: int = 128,
    sps: int = 4,
    rolloff: float = 0.35,
    seed: int = 0,
    return_snr: bool = False,
):
    """生成一个 IQ 数据集。

    Returns
    -------
    X : (N, 2, L) float32，通道 0=I、1=Q
    y : (N,) int64
    snr : (N,) float32（仅当 return_snr=True）
    """
    rng = np.random.default_rng(seed)
    mods = list(modulations)
    consts = {m: make_constellation(m) for m in mods}
    L = n_symbols * sps

    X = np.empty((n_samples, 2, L), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int64)
    snrs = np.empty(n_samples, dtype=np.float32)

    # 预生成 CFG 的 SNR 采样
    for i in range(n_samples):
        k = int(rng.integers(0, len(mods)))
        mod = mods[k]
        pts = consts[mod]
        syms = pts[rng.integers(0, pts.size, size=n_symbols)]
        wave = pulse_shape(syms, sps, rolloff)
        if wave.size < L:
            wave = np.pad(wave, (0, L - wave.size))
        wave = wave[:L]

        snr_i = float(rng.uniform(cfg.snr_db[0], cfg.snr_db[1]))
        wave = apply_channel(wave, cfg, rng, snr_db=snr_i)

        X[i, 0] = wave.real.astype(np.float32)
        X[i, 1] = wave.imag.astype(np.float32)
        y[i] = k
        snrs[i] = snr_i

    if return_snr:
        return X, y, snrs
    return X, y
