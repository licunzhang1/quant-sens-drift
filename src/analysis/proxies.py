"""Proxy quantities: can a quantity that needs **no test set** predict
quantization sensitivity?

Motivation
----------
If a cheap quantity independent of the deployment domain (weight kurtosis, say)
could predict the sensitivity ranking, then the cost of "not knowing the
deployment domain" could be partly removed -- a strong bonus.

Candidate proxies (statistics of the weights or activations only; no labels and
no target-domain data needed)
------------------------------------------------------------------------------
kurtosis       : excess kurtosis. The peakier the distribution (heavy tails),
                 the more a few outliers dominate the scale, and the more
                 quantization loses
peak_to_rms    : max|w| / rms(w). Larger means the dynamic range is more
                 concentrated
dyn_range_db   : dynamic range (dB)
level_usage    : fraction of the quantization levels actually used at a given
                 bit width (lower means the scale is held hostage by outliers)
eff_bits       : effective bit width = log2(dynamic range / quantization step),
                 "how many bits are actually used"
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
    """Excess kurtosis (0 for a normal distribution)."""
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
    """Fraction of the non-zero quantization levels used at a given bit width.

    Closer to 1 means the scale fits the tensor better.
    """
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
    """Effective bit width: dynamic range / noise floor decides how many levels
    can actually be resolved."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        return float("nan")
    hi = np.max(np.abs(x))
    if hi < eps:
        return float("nan")
    # use the scale of adjacent differences as a proxy for the smallest
    # resolvable step
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
    """All proxy quantities of a single tensor."""
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
    """``{layer: tensor}`` -> ``({proxy_name: (n_layers,)}, layer_names)``."""
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
    """Correlate every proxy quantity with the sensitivity (Spearman with a
    permutation test).

    Returns a list sorted by descending |rho|.
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
