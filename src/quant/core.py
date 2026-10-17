"""Quantization math core -- pure numpy, no torch dependency.

Design principle
----------------
This layer is the scientific core of the whole project (quantization,
reconstruction error, sensitivity metrics) and **must be verifiable
independently of any deep-learning framework**.
The torch-side ``fake_quant.py`` merely moves this mathematics onto tensors; the
two have to agree semantically.

Conventions
-----------
* ``x`` is a float array of arbitrary shape.
* ``bits`` is the bit width. Symmetric quantization is meaningless at
  ``bits=1`` (qmax=0) and raises an error.
* ``per_channel_axis=None`` means per-tensor; an integer means the parameters
  are computed per channel along that axis. Negative axes follow numpy rules.
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
# quantization range
# --------------------------------------------------------------------------
def qrange(bits: int, symmetric: bool = True) -> Tuple[int, int]:
    """Return the integer code range ``(qmin, qmax)`` for a given bit width."""
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    if symmetric:
        if bits < 2:
            raise ValueError(
                "symmetric quantization is meaningless at bits=1 (qmax would "
                "be 0). Use asymmetric, or raise the bit width to >= 2."
            )
        qmax = 2 ** (bits - 1) - 1
        return -qmax, qmax
    return 0, 2 ** bits - 1


# --------------------------------------------------------------------------
# quantization parameters
# --------------------------------------------------------------------------
def _reduce_axes(ndim: int, axis: Optional[int]) -> Optional[Tuple[int, ...]]:
    """Axes to reduce over in per-channel mode (everything but ``axis``).

    Returns None when ``axis=None`` (reduce over everything).
    """
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
    """Compute ``(scale, zero_point)`` with shapes that broadcast onto ``x``.

    * symmetric: ``scale = max|x| / qmax``, ``zero_point = 0``
    * asymmetric: ``scale = (max-min)/(2^bits-1)``, ``zero_point = round(-min/scale)``
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("cannot quantize an empty array")
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
# quantize / dequantize
# --------------------------------------------------------------------------
def quantize(
    x: np.ndarray,
    bits: int,
    symmetric: bool = True,
    per_channel_axis: Optional[int] = None,
    scale: Optional[np.ndarray] = None,
    zero: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return integer codes (stored as float64 for downstream arithmetic)."""
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
    """Turn integer codes back into floats."""
    return (np.asarray(q, dtype=np.float64) - zero) * scale


def quantize_dequantize(
    x: np.ndarray,
    bits: int,
    symmetric: bool = True,
    per_channel_axis: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The most common entry point: quantize, then immediately dequantize.

    Returns ``(xq, scale, zero)``. This is "fake quantization": the values stay
    float, but they already carry the quantization error.
    """
    x = np.asarray(x, dtype=np.float64)
    scale, zero = quant_params(x, bits, symmetric, per_channel_axis)
    q = quantize(x, bits, symmetric, per_channel_axis, scale, zero)
    return dequantize(q, scale, zero), scale, zero


# --------------------------------------------------------------------------
# reconstruction error
# --------------------------------------------------------------------------
def reconstruction_error(x: np.ndarray, xq: np.ndarray) -> Dict[str, float]:
    """Reconstruction error introduced by quantization.

    Returns
    -------
    mse      : mean squared error
    nmse     : normalized MSE ``||x-xq||^2 / ||x||^2`` (the main metric,
               comparable across layers)
    max_abs  : maximum absolute error
    snr_db   : reconstruction SNR derived from the NMSE
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
    """Sweep one tensor across the bit-width ladder.

    Returns ``{bits: reconstruction_error}``. This is the input of the "equal
    reconstruction error arm": each error curve ``eps_l(b)``.
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
    """Compute the NMSE per channel, then take the **unweighted** mean.

    Why it is needed
    ----------------
    The global NMSE is energy-weighted: if one channel of a weight matrix has a
    scale 1000x larger than the others, the global NMSE is decided entirely by
    it, and a small-scale channel being "destroyed by quantization" is invisible
    in the number.

    In quantization analysis, however, what matters is "**is each channel
    represented sensibly**", so the channels should count equally. This metric
    is more sensitive to per-channel vs per-tensor comparisons and closer to the
    actual concern.
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
    """``{layer: {bits: {...}}}`` -> ``(E, layer_names)`` with ``E`` shaped
    ``(n_layers, n_bits)``."""
    names = list(per_layer_curves.keys())
    E = np.full((len(names), len(bits_ladder)), np.nan, dtype=np.float64)
    for i, name in enumerate(names):
        for j, b in enumerate(bits_ladder):
            rec = per_layer_curves[name].get(int(b))
            if rec is not None:
                E[i, j] = rec[metric]
    return E, names
