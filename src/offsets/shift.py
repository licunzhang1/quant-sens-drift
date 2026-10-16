"""分布偏移的构造与管理。

当前偏移的**参数**都在 ``configs/base.yaml`` 的 ``data.shifts`` 里，
本模块负责把它们整理成可迭代的对象，并提供组合/强度缩放等便利操作。

为什么偏移要配置化
------------------
本项目的核心问题是"域偏移如何影响量化敏感性"，所以**偏移是自变量**。
它必须像超参一样被记录、可复现、可批量扫描 —— 不能写死在代码里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from ..data.synth import ChannelConfig

__all__ = ["ShiftSpec", "list_shifts", "build_shift", "scale_shift", "describe_all"]


@dataclass(frozen=True)
class ShiftSpec:
    """一个具名偏移。"""

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
    """把配置里的 ``data.shifts`` 转成 ``ShiftSpec`` 列表。"""
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
    raise KeyError(f"未知偏移 {name!r}；可用: {avail}")


def scale_shift(spec: ShiftSpec, factor: float) -> ShiftSpec:
    """按比例缩放偏移强度（用于"偏移强度连续变化"的扫描实验）。

    只缩放非零的损伤项与 SNR 的**偏离量**，保持参考 SNR 不变。
    """
    c = spec.channel
    ref_snr = 14.0  # 参考点：偏离它多少，缩放后再加回去
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
