"""Construction and management of distribution shifts.

The current shift **parameters** all live under ``data.shifts`` in
``configs/base.yaml``. This module turns them into iterable objects and offers
convenience operations such as composition and intensity scaling.

Why shifts are configurable
---------------------------
The core question of this project is "how does domain shift affect quantization
sensitivity", so **the shift is the independent variable**. It has to be
recorded, reproducible and batch-scannable like any hyperparameter -- it cannot
be hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from ..data.synth import ChannelConfig

__all__ = ["ShiftSpec", "list_shifts", "build_shift", "scale_shift", "describe_all"]


@dataclass(frozen=True)
class ShiftSpec:
    """A named shift."""

    name: str
    channel: ChannelConfig
    kind: str = ""          # snr | cfo | gain | phase_noise | iq_imbalance | mixed
    note: str = ""

    def describe(self) -> str:
        return f"{self.name:<16s} [{self.kind:<14s}] {self.channel.describe()}"


def _infer_kind(cfg: Dict) -> str:
    keys = {k for k, v in cfg.items()
            if k != "snr_db" and isinstance(v, (int, float)) and v != 0}
    if not keys:
        return "snr"
    if len(keys) > 1:
        return "mixed"
    return next(iter(keys))


def list_shifts(shifts_cfg: Optional[Dict]) -> List[ShiftSpec]:
    """Turn ``data.shifts`` from the config into a list of ``ShiftSpec``."""
    out: List[ShiftSpec] = []
    for name, cfg in (shifts_cfg or {}).items():
        cfg = dict(cfg)
        out.append(ShiftSpec(name=name,
                             channel=ChannelConfig.from_dict(cfg),
                             kind=_infer_kind(cfg)))
    return out


def build_shift(shifts_cfg: Optional[Dict], name: str) -> ShiftSpec:
    for s in list_shifts(shifts_cfg):
        if s.name == name:
            return s
    avail = [s.name for s in list_shifts(shifts_cfg)]
    raise KeyError(f"unknown shift {name!r}; available: {avail}")


def scale_shift(spec: ShiftSpec, factor: float) -> ShiftSpec:
    """Scale the shift intensity proportionally (used for sweeps in which the
    "shift intensity varies continuously").

    Only the non-zero impairments and the **deviation** of the SNR are scaled;
    the reference SNR is kept unchanged.
    """
    c = spec.channel
    ref_snr = 14.0  # reference point: scale the deviation away from it, then add it back
    lo = ref_snr + (c.snr_db[0] - ref_snr) * factor
    hi = ref_snr + (c.snr_db[1] - ref_snr) * factor
    new = ChannelConfig(
        snr_db=(lo, hi),
        cfo=c.cfo * factor,
        phase_noise_deg=c.phase_noise_deg * factor,
        iq_gain_db=c.iq_gain_db * factor,
        iq_phase_deg=c.iq_phase_deg * factor,
        gain_db=c.gain_db * factor,
    )
    return ShiftSpec(name=f"{spec.name}@x{factor:g}", channel=new, kind=spec.kind,
                     note=f"scaled x{factor:g} of {spec.name}")


def describe_all(shifts_cfg: Optional[Dict]) -> Iterator[str]:
    for s in list_shifts(shifts_cfg):
        yield s.describe()
