"""量化方案的定义与枚举。

一个"方案"= 量化什么（权重/激活）× 粒度（per-tensor / per-channel）× 对称性 × 位宽。

实验里的因子 A 就是这里的 ``QuantScheme`` 集合。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

__all__ = ["QuantScheme", "default_scheme_grid", "scheme_from_name"]


@dataclass(frozen=True)
class QuantScheme:
    """一种量化配置。

    Attributes
    ----------
    name : 人类可读的名字，会进结果表
    bits : 位宽
    symmetric : 对称 / 非对称
    quantize_weights : 是否量化权重
    quantize_activations : 是否量化激活
    weight_per_channel : 权重是否 per-channel（沿输出通道）
    activation_per_channel : 激活是否 per-channel（一般 False）
    """

    name: str
    bits: int
    symmetric: bool = True
    quantize_weights: bool = True
    quantize_activations: bool = False
    weight_per_channel: bool = True
    activation_per_channel: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def is_weight_only(self) -> bool:
        return self.quantize_weights and not self.quantize_activations

    @property
    def is_weight_and_activation(self) -> bool:
        return self.quantize_weights and self.quantize_activations


def make(
    name: str,
    bits: int,
    symmetric: bool = True,
    w: bool = True,
    a: bool = False,
    w_pc: bool = True,
    a_pc: bool = False,
) -> QuantScheme:
    return QuantScheme(name, bits, symmetric, w, a, w_pc, a_pc)


def default_scheme_grid(bits_ladder=(2, 3, 4, 6, 8)) -> List[QuantScheme]:
    """默认方案格子（因子 A）。

    为了控制实验规模，只枚举有意义的组合：
      * 仅权重（per-tensor / per-channel） × 位宽
      * 权重 + 激活（对称 / 非对称） × 位宽
    """
    schemes: List[QuantScheme] = []
    for b in bits_ladder:
        schemes.append(make(f"W{b}_pc", b, True, w=True, a=False, w_pc=True))
        schemes.append(make(f"W{b}_pt", b, True, w=True, a=False, w_pc=False))
    for b in bits_ladder:
        if b >= 3 or True:
            schemes.append(make(f"WA{b}_sym", b, True, w=True, a=True, w_pc=True, a_pc=False))
        schemes.append(make(f"WA{b}_asym", b, False, w=True, a=True, w_pc=True, a_pc=False))
    return schemes


def scheme_from_name(name: str, bits_ladder=(2, 3, 4, 6, 8)) -> Optional[QuantScheme]:
    for s in default_scheme_grid(bits_ladder):
        if s.name == name:
            return s
    return None
