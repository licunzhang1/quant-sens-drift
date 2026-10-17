"""Classical AMC features: higher-order cumulants plus envelope statistics.

Why the project needs them
--------------------------
Two uses, one implementation:

1. **Input representation of the reference model** (``tests/smoke_pipeline.py``).
   In a raw IQ waveform the modulation information lives in the **distribution**
   rather than at a few sample points, and a plain MLP struggles to learn it on
   its own (measured: 1.0 train / 0.50 test, with only BPSK separable).
   With cumulant features the model generalises properly, and only then does the
   "domain gap" mean anything.

2. **Classical DSP baseline arm** (cut 2 in the study design).
   It answers "how much better is deep learning than the classical approach" --
   a question the literature asserts widely but rarely tests with an **equally
   strict split protocol**.

Cumulant definitions (complex random variable z, zero mean)
-----------------------------------------------------------
C20 = E[z^2]
C21 = E[|z|^2]
C40 = E[z^4] - 3*E[z^2]^2
C41 = E[z^3 z*] - 3*E[z^2]*E[|z|^2]
C42 = E[|z|^4] - |E[z^2]|^2 - 2*E[|z|^2]^2
C60 = E[z^6] - 15*E[z^2]E[z^4] + 30*E[z^2]^3
C63 = E[|z|^6] - 9*E[|z|^4]E[|z|^2] - 6*E[z^2*]... (see the implementation, which
      uses the common approximate form)

These quantities are sensitive to the constellation shape and naturally have
some invariance to rotation and scale, which makes them the standard classical
AMC features.
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
    """Phase-histogram entropy (normalized to [0, 1]).

    BPSK has the most concentrated phase distribution.
    """
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
    """Turn a ``(N, 2, L)`` IQ batch into an ``(N, F)`` feature matrix.

    Every sample point along ``L`` takes part in the statistics (pulse-shaping
    correlation makes them dependent, but the empirical cumulants still
    discriminate -- this is standard practice in classical AMC).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"input should be (N, 2, L), got {x.shape}")
    n = x.shape[0]
    z = x[:, 0, :] + 1j * x[:, 1, :]  # (N, L)

    # normalize each sample to unit power individually for scale invariance
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
