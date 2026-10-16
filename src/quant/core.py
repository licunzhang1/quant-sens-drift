"""量化数学核心 —— 纯 numpy，不依赖 torch。

设计原则
--------
这一层是整个项目的科学核心（量化、重构误差、敏感性度量），
**必须能独立于深度学习框架被验证**。
torch 侧的 ``fake_quant.py`` 只是把这里的数学搬到张量上，两者语义必须一致。

约定
----
* ``x`` 是 float 数组，任意形状。
* ``bits`` 是位宽。对称量化在 ``bits=1`` 时无意义（qmax=0），会抛错。
* ``per_channel_axis=None`` 表示 per-tensor；给整数表示沿该轴逐通道求参数。
  负轴按 numpy 规则解释。
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "qrange",
    "quant_params",
    "quantize",
    "dequantize",
    "quantize_dequantize",
    "reconstruction_error",
    "channel_wise_nmse",
    "quant_error_curve",
    "DEFAULT_BITS_LADDER",
]

DEFAULT_BITS_LADDER: Tuple[int, ...] = (2, 3, 4, 6, 8)

_EPS = 1e-12


# --------------------------------------------------------------------------
# 量化区间
# --------------------------------------------------------------------------
def qrange(bits: int, symmetric: bool = True) -> Tuple[int, int]:
    """返回给定位宽下的整数码区间 ``(qmin, qmax)``。"""
    if bits < 1:
        raise ValueError(f"bits 必须 >= 1，收到 {bits}")
    if symmetric:
        if bits < 2:
            raise ValueError(
                "对称量化在 bits=1 时无意义（qmax 会等于 0）。"
                "请用 asymmetric，或把位宽提到 >= 2。"
            )
        qmax = 2 ** (bits - 1) - 1
        return -qmax, qmax
    return 0, 2 ** bits - 1


# --------------------------------------------------------------------------
# 量化参数
# --------------------------------------------------------------------------
def _reduce_axes(ndim: int, axis: Optional[int]) -> Optional[Tuple[int, ...]]:
    """per-channel 时要归约的轴（除 axis 外全部）。axis=None 时返回 None（全归约）。"""
    if axis is None:
        return None
    ax = axis % ndim
    return tuple(i for i in range(ndim) if i != ax)


def quant_params(
    x: np.ndarray,
    bits: int,
    symmetric: bool = True,
    per_channel_axis: Optional[int] = None,
    eps: float = _EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算 ``(scale, zero_point)``，形状可广播到 ``x``。

    * 对称：``scale = max|x| / qmax``，``zero_point = 0``
    * 非对称：``scale = (max-min)/(2^bits-1)``，``zero_point = round(-min/scale)``
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("空数组无法量化")
    axes = _reduce_axes(x.ndim, per_channel_axis)

    if symmetric:
        qmin, qmax = qrange(bits, symmetric=True)
        amax = np.max(np.abs(x), axis=axes, keepdims=True)
        scale = np.maximum(amax, eps) / qmax
        zero = np.zeros_like(scale)
    else:
        qmin, qmax = qrange(bits, symmetric=False)
        xmin = np.min(x, axis=axes, keepdims=True)
        xmax = np.max(x, axis=axes, keepdims=True)
        scale = np.maximum(xmax - xmin, eps) / qmax
        zero = np.round(-xmin / scale)
        np.clip(zero, qmin, qmax, out=zero)

    return scale, zero


# --------------------------------------------------------------------------
# 量化 / 反量化
# --------------------------------------------------------------------------
def quantize(
    x: np.ndarray,
    bits: int,
    symmetric: bool = True,
    per_channel_axis: Optional[int] = None,
    scale: Optional[np.ndarray] = None,
    zero: Optional[np.ndarray] = None,
) -> np.ndarray:
    """返回整数码（float64 存储，便于后续运算）。"""
    x = np.asarray(x, dtype=np.float64)
    if scale is None or zero is None:
        scale, zero = quant_params(x, bits, symmetric, per_channel_axis)
    qmin, qmax = qrange(bits, symmetric)
    q = np.round(x / scale + zero)
    return np.clip(q, qmin, qmax)


def dequantize(
    q: np.ndarray,
    scale: np.ndarray,
    zero: np.ndarray,
) -> np.ndarray:
    """整数码还原为浮点。"""
    return (np.asarray(q, dtype=np.float64) - zero) * scale


def quantize_dequantize(
    x: np.ndarray,
    bits: int,
    symmetric: bool = True,
    per_channel_axis: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """最常用的入口：量化后立刻反量化，返回 ``(xq, scale, zero)``。

    这是 "fake quantization"：数值上保留 float，但已经带上了量化误差。
    """
    x = np.asarray(x, dtype=np.float64)
    scale, zero = quant_params(x, bits, symmetric, per_channel_axis)
    q = quantize(x, bits, symmetric, per_channel_axis, scale, zero)
    return dequantize(q, scale, zero), scale, zero


# --------------------------------------------------------------------------
# 重构误差
# --------------------------------------------------------------------------
def reconstruction_error(x: np.ndarray, xq: np.ndarray) -> Dict[str, float]:
    """量化带来的重构误差。

    返回
    ----
    mse      : 均方误差
    nmse     : 归一化均方误差 ``||x-xq||^2 / ||x||^2``（跨层可比的主指标）
    max_abs  : 最大绝对误差
    snr_db   : 由 NMSE 换算的重构信噪比
    """
    x = np.asarray(x, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    diff = x - xq
    mse = float(np.mean(diff ** 2))
    denom = float(np.mean(x ** 2))
    nmse = mse / (denom + _EPS)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    snr_db = float(10.0 * np.log10(1.0 / (nmse + _EPS)))
    return {"mse": mse, "nmse": nmse, "max_abs": max_abs, "snr_db": snr_db}


def quant_error_curve(
    x: np.ndarray,
    bits_ladder: Sequence[int] = DEFAULT_BITS_LADDER,
    symmetric: bool = True,
    per_channel_axis: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """对同一个张量扫一遍位宽阶梯，返回 ``{bits: reconstruction_error}``。

    这是"等重构误差臂"的输入：每条误差曲线 ``eps_l(b)``。
    """
    out: Dict[int, Dict[str, float]] = {}
    for b in bits_ladder:
        if symmetric and b < 2:
            continue
        xq, _, _ = quantize_dequantize(x, b, symmetric, per_channel_axis)
        out[int(b)] = reconstruction_error(x, xq)
    return out


def channel_wise_nmse(
    x: np.ndarray,
    xq: np.ndarray,
    axis: int = 0,
    eps: float = _EPS,
) -> float:
    """逐通道算 NMSE，再取**无权**平均。

    为什么需要它
    ------------
    全局 NMSE 是按能量加权的：如果权重矩阵里某个通道的尺度比别人大 1000 倍，
    全局 NMSE 就完全由它决定，小尺度通道被"量化摧毁"这件事在数字上看不出来。

    但在量化分析里我们关心的是"**每个通道是否被合理表示**"，各通道应当等权。
    这个度量对 per-channel vs per-tensor 的对比更敏感，也更贴近实际关切。
    """
    x = np.asarray(x, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    if x.ndim < 2:
        return reconstruction_error(x, xq)["nmse"]
    ax = axis % x.ndim
    axes = tuple(i for i in range(x.ndim) if i != ax)
    num = np.mean((x - xq) ** 2, axis=axes)
    den = np.mean(x ** 2, axis=axes)
    return float(np.mean(num / (den + eps)))



def error_curves_to_matrix(
    per_layer_curves: Dict[str, Dict[int, Dict[str, float]]],
    metric: str = "nmse",
    bits_ladder: Sequence[int] = DEFAULT_BITS_LADDER,
) -> Tuple[np.ndarray, list]:
    """``{layer: {bits: {...}}}`` → ``(E, layer_names)``，``E`` 形状 ``(n_layers, n_bits)``。"""
    names = list(per_layer_curves.keys())
    E = np.full((len(names), len(bits_ladder)), np.nan, dtype=np.float64)
    for i, name in enumerate(names):
        for j, b in enumerate(bits_ladder):
            rec = per_layer_curves[name].get(int(b))
            if rec is not None:
                E[i, j] = rec[metric]
    return E, names
