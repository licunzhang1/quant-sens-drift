"""代理量：能不能用**不需要测试集**的量预测量化敏感性？

动机
----
如果存在一个廉价的、与部署域无关的量（例如权重峰度）可以预测敏感性排序，
那么"不知道部署域"的代价就可以被部分消掉 —— 这是很强的加分项。

候选代理（全部只看权重或激活的统计量，不需要标签、不需要目标域数据）
------------------------------------------------------------------
kurtosis       : 超额峰度。分布越尖（长尾），少量离群值支配 scale，量化越吃亏
peak_to_rms    : max|w| / rms(w)。越大说明动态范围越集中
dyn_range_db   : 动态范围（dB）
level_usage    : 给定位宽下实际被用到的量化级比例（越低说明 scale 被离群值绑架）
eff_bits       : 有效位宽 = log2(动态范围/量化步长)，"实际用上几 bit"
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from ..quant.core import quant_params, quantize_dequantize

__all__ = [
    "excess_kurtosis",
    "peak_to_rms",
    "dynamic_range_db",
    "level_usage",
    "effective_bits",
    "proxy_vector",
    "proxy_matrix",
    "correlate_with_sensitivity",
]

_EPS = 1e-12


def excess_kurtosis(x: np.ndarray) -> float:
    """超额峰度（正态分布为 0）。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 4:
        return float("nan")
    m = x.mean()
    s = x.std()
    if s < _EPS:
        return float("nan")
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def peak_to_rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return float("nan")
    rms = np.sqrt(np.mean(x ** 2))
    if rms < _EPS:
        return float("nan")
    return float(np.max(np.abs(x)) / rms)


def dynamic_range_db(x: np.ndarray, eps: float = _EPS) -> float:
    x = np.abs(np.asarray(x, dtype=np.float64).ravel())
    if x.size == 0:
        return float("nan")
    hi = np.max(x)
    lo = np.maximum(np.min(x[x > 0]) if np.any(x > 0) else eps, eps)
    if hi < eps:
        return float("nan")
    return float(20.0 * np.log10(hi / lo))


def level_usage(x: np.ndarray, bits: int, symmetric: bool = True) -> float:
    """给定位宽下，非零量化级被用到的比例。越接近 1 说明 scale 适配得越好。"""
    from ..quant.core import qrange, quantize

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    scale, zero = quant_params(x, bits, symmetric)
    q = quantize(x, bits, symmetric, None, scale, zero)
    qmin, qmax = qrange(bits, symmetric)
    levels = int(qmax - qmin + 1)
    used = np.unique(np.round(q).astype(np.int64)).size
    return float(used / levels)


def effective_bits(x: np.ndarray, eps: float = _EPS) -> float:
    """有效位宽：动态范围 / 噪声底 决定实际能分辨多少级。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        return float("nan")
    hi = np.max(np.abs(x))
    if hi < eps:
        return float("nan")
    # 用相邻差分的尺度作为"可分辨最小步长"的代理
    xs = np.sort(x)
    d = np.diff(xs)
    d = d[d > eps]
    if d.size == 0:
        return 0.0
    step = np.median(d)
    if step < eps:
        return float("nan")
    return float(np.log2(2 * hi / step))


def proxy_vector(x: np.ndarray, bits: int = 4, symmetric: bool = True) -> Dict[str, float]:
    """单个张量的全部代理量。"""
    return {
        "kurtosis": excess_kurtosis(x),
        "peak_to_rms": peak_to_rms(x),
        "dyn_range_db": dynamic_range_db(x),
        "level_usage": level_usage(x, bits, symmetric),
        "eff_bits": effective_bits(x),
        "absmax": float(np.max(np.abs(x))) if np.size(x) else float("nan"),
        "std": float(np.std(x)) if np.size(x) else float("nan"),
        "n_params": int(np.size(x)),
    }


def proxy_matrix(
    layer_tensors: Mapping[str, np.ndarray],
    bits: int = 4,
    symmetric: bool = True,
    keys: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """``{layer: tensor}`` → ``({proxy_name: (n_layers,)}, layer_names)``。"""
    names = list(layer_tensors.keys())
    rows = [proxy_vector(np.asarray(layer_tensors[n]), bits, symmetric) for n in names]
    if not rows:
        return {}, names
    all_keys = list(keys) if keys is not None else list(rows[0].keys())
    out: Dict[str, np.ndarray] = {}
    for k in all_keys:
        out[k] = np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)
    return out, names


def correlate_with_sensitivity(
    proxies: Mapping[str, np.ndarray],
    sensitivity: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """把每个代理量与敏感性做 Spearman 相关（含置换检验）。

    返回按 |ρ| 降序排列的列表。
    """
    from .rank_corr import permutation_test

    s = np.asarray(sensitivity, dtype=np.float64).ravel()
    rows: List[Dict[str, object]] = []
    for name, vals in proxies.items():
        v = np.asarray(vals, dtype=np.float64).ravel()
        if v.size != s.size:
            continue
        rho, p = permutation_test(v, s, n_permutations=n_permutations, seed=seed)
        rows.append(
            {
                "proxy": name,
                "rho": round(float(rho), 4) if np.isfinite(rho) else float("nan"),
                "p_value": round(float(p), 6) if np.isfinite(p) else float("nan"),
                "significant": bool(np.isfinite(p) and p < 0.05),
                "abs_rho": abs(float(rho)) if np.isfinite(rho) else 0.0,
            }
        )
    rows.sort(key=lambda r: r["abs_rho"], reverse=True)
    return rows
