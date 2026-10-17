"""Definition and enumeration of quantization schemes.

A "scheme" = what is quantized (weights/activations) x granularity
(per-tensor / per-channel) x symmetry x bit width.

Factor A in the experiments is the set of ``QuantScheme`` objects defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

__all__ = ["QuantScheme", "default_scheme_grid", "scheme_from_name"]


@dataclass(frozen=True)
class QuantScheme:
    """One quantization configuration.

    Attributes
    ----------
    name : human-readable name, goes into the results table
    bits : bit width
    symmetric : symmetric / asymmetric
    quantize_weights : whether to quantize the weights
    quantize_activations : whether to quantize the activations
    weight_per_channel : whether weights are per-channel (along the output axis)
    activation_per_channel : whether activations are per-channel (usually False)
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
    """The default scheme grid (factor A).

    To keep the experiment size manageable, only meaningful combinations are
    enumerated:
      * weights only (per-tensor / per-channel) x bit width
      * weights + activations (symmetric / asymmetric) x bit width
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
