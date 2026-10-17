"""Synthetic IQ signal generation -- zero downloads, fully controllable shifts.

Why synthetic data by default
-----------------------------
This project studies **quantization sensitivity under distribution shift**, so
the kind and strength of the shift must be **controlled exactly**. Public
datasets (RML2018.01A and friends) usually provide a single fixed channel
condition and cannot support controlled shift experiments. Synthetic data has
two further advantages: zero downloads, reproducible, and unlimited sampling.

To plug in a public dataset, implement a new ``Dataset``; this module does not
need to change.

The sample format matches RadioML: ``X: (N, 2, L)`` with channel 0=I and 1=Q.
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
# constellations
# --------------------------------------------------------------------------
def _pam_levels(m: int) -> np.ndarray:
    """Normalized m-PAM levels (unit average energy)."""
    levels = np.arange(-(m - 1), m, 2, dtype=np.float64)
    return levels / np.sqrt(np.mean(levels ** 2))


def make_constellation(name: str) -> np.ndarray:
    """Return complex constellation points with unit average energy."""
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
        lv = _pam_levels(4)[1::2]  # magnitude of order 1/sqrt(2)
        outer = np.array([a + 1j * b for a in lv for b in lv], dtype=np.complex128)
        lv2 = _pam_levels(2)
        inner = np.array([a + 1j * b for a in lv2 for b in lv2], dtype=np.complex128)
        pts = np.concatenate([outer, inner])
    else:
        raise ValueError(f"unknown modulation: {name}")
    pts = pts.astype(np.complex128)
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


MODULATIONS: Tuple[str, ...] = ("bpsk", "qpsk", "8psk", "8qam", "16qam")


# --------------------------------------------------------------------------
# pulse shaping
# --------------------------------------------------------------------------
def rrc_filter(beta: float, sps: int, span: int = 6) -> np.ndarray:
    """Root-raised-cosine filter (unit energy)."""
    if sps < 1:
        raise ValueError("sps must be >= 1")
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
        raise ValueError("degenerate RRC filter energy")
    return h / np.sqrt(e)


def pulse_shape(symbols: np.ndarray, sps: int, beta: float, span: int = 6) -> np.ndarray:
    """Upsample, then apply RRC pulse shaping."""
    up = np.zeros(symbols.size * sps, dtype=np.complex128)
    up[::sps] = symbols
    h = rrc_filter(beta, sps, span)
    return np.convolve(up, h, mode="same")


# --------------------------------------------------------------------------
# channel
# --------------------------------------------------------------------------
@dataclass
class ChannelConfig:
    """Channel impairment config. Domain A and Domain B each use one."""

    snr_db: Tuple[float, float] = (10.0, 18.0)   # interval, drawn per sample
    cfo: float = 0.0                             # normalized carrier frequency offset
    phase_noise_deg: float = 0.0                 # phase-noise standard deviation (deg)
    iq_gain_db: float = 0.0                      # IQ gain imbalance
    iq_phase_deg: float = 0.0                    # IQ phase imbalance (deg)
    gain_db: float = 0.0                         # overall receiver gain offset

    @classmethod
    def from_dict(cls, d: Optional[Dict] = None) -> "ChannelConfig":
        d = dict(d or {})
        snr = d.pop("snr_db", (10.0, 18.0))
        if np.isscalar(snr):
            snr = (float(snr), float(snr))
        return cls(snr_db=(float(snr[0]), float(snr[1])), **d)

    def describe(self) -> str:
        return (
            f"SNR in [{self.snr_db[0]:g},{self.snr_db[1]:g}]dB"
            f", CFO={self.cfo:g}, PN={self.phase_noise_deg:g} deg"
            f", IQg={self.iq_gain_db:g}dB, IQphi={self.iq_phase_deg:g} deg"
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
    """Apply the channel impairments given by the config.

    Order: IQ imbalance -> CFO -> phase noise -> gain -> AWGN.
    """
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
    # normalize to unit average power throughout (consistent with RadioML)
    p = np.mean(np.abs(y) ** 2)
    if p > 1e-20:
        y = y / np.sqrt(p)
    return y


# --------------------------------------------------------------------------
# datasets
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
    """Generate one IQ dataset.

    Returns
    -------
    X : (N, 2, L) float32, channel 0=I and 1=Q
    y : (N,) int64
    snr : (N,) float32 (only when return_snr=True)
    """
    rng = np.random.default_rng(seed)
    mods = list(modulations)
    consts = {m: make_constellation(m) for m in mods}
    L = n_symbols * sps

    X = np.empty((n_samples, 2, L), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int64)
    snrs = np.empty(n_samples, dtype=np.float32)

    # draw the SNR samples declared by the config
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
