"""Layer-wise quantization sensitivity matrix.

Core quantity
-------------
``Delta_l(b) = M(quantizing layer l to b bits) - M(full float)``

where M is the evaluation metric (accuracy or loss).

The design is **framework-independent**: it depends only on two callbacks
  * ``evaluate()``             -> float, the metric at full precision
  * ``quantize_layer(name, b)`` -> context manager that temporarily quantizes
    layer l

That way the numpy smoke test and the real torch training can share one
implementation.
"""

from __future__ import annotations

import json
import pathlib
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ContextManager, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "SensitivityResult",
    "compute_sensitivity_matrix",
    "sensitivity_at_bits",
    "normalise_sensitivity",
]


@dataclass
class SensitivityResult:
    """Layer-wise sensitivity matrix and its metadata."""

    layer_names: List[str]
    bits_ladder: List[int]
    delta: np.ndarray          # (n_layers, n_bits)  Deltametric, oriented so larger = worse
    absolute: np.ndarray       # (n_layers, n_bits)  absolute metric after quantization
    baseline: float            # full-precision metric
    metric: str
    higher_is_better: bool
    domain: str = ""
    backbone: str = ""
    elapsed_s: float = 0.0
    extra: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------ convenience
    @property
    def n_layers(self) -> int:
        return len(self.layer_names)

    def sensitivity_at(self, bits: int) -> np.ndarray:
        """Sensitivity vector at a given bit width.

        Rank analysis normally uses the reference bit width.
        """
        if bits not in self.bits_ladder:
            raise KeyError(f"{bits} is not in the bit-width ladder {self.bits_ladder}")
        return self.delta[:, self.bits_ladder.index(bits)]

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer_names": list(self.layer_names),
            "bits_ladder": list(self.bits_ladder),
            "delta": self.delta.tolist(),
            "absolute": self.absolute.tolist(),
            "baseline": self.baseline,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "domain": self.domain,
            "backbone": self.backbone,
            "elapsed_s": self.elapsed_s,
            **self.extra,
        }

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            from ..utils import dump_json

            dump_json(path, self.to_dict())
        else:
            np.savez_compressed(
                path,
                delta=self.delta,
                absolute=self.absolute,
                baseline=np.array([self.baseline]),
                layer_names=np.array(self.layer_names, dtype=object),
                bits_ladder=np.array(self.bits_ladder),
                metric=np.array([self.metric]),
                domain=np.array([self.domain]),
                backbone=np.array([self.backbone]),
            )
        return path

    @staticmethod
    def load(path: str | Path) -> "SensitivityResult":
        path = Path(path)
        with np.load(path, allow_pickle=True) as z:
            return SensitivityResult(
                layer_names=[str(s) for s in z["layer_names"]],
                bits_ladder=[int(b) for b in z["bits_ladder"]],
                delta=z["delta"],
                absolute=z["absolute"],
                baseline=float(z["baseline"][0]),
                metric=str(z["metric"][0]),
                higher_is_better=True,
                domain=str(z["domain"][0]) if "domain" in z else "",
                backbone=str(z["backbone"][0]) if "backbone" in z else "",
            )


def compute_sensitivity_matrix(
    layer_names: Sequence[str],
    bits_ladder: Sequence[int],
    evaluate: Callable[[], float],
    quantize_layer: Callable[[str, int], ContextManager],
    *,
    metric: str = "acc",
    higher_is_better: bool = True,
    domain: str = "",
    backbone: str = "",
    progress: bool = True,
) -> SensitivityResult:
    """Main entry point: compute the full layer-wise sensitivity matrix.

    Parameters
    ----------
    layer_names : layers to analyse (the order is the row order of the output)
    bits_ladder : bit-width ladder, e.g. (2, 3, 4, 6, 8)
    evaluate : zero-argument function returning the current model's metric on
        the target domain
    quantize_layer : ``(layer_name, bits) -> context manager``
    higher_is_better : with True (accuracy), Delta is defined as the drop; with
        False (loss), as the increase. Either way **larger Delta = worse**, so the
        values can be compared and ranked directly.

    Returns
    -------
    SensitivityResult
    """
    t0 = time.time()
    bits_ladder = [int(b) for b in bits_ladder]
    layer_names = [str(s) for s in layer_names]

    baseline = float(evaluate())
    n_l, n_b = len(layer_names), len(bits_ladder)
    delta = np.full((n_l, n_b), np.nan, dtype=np.float64)
    absolute = np.full((n_l, n_b), np.nan, dtype=np.float64)

    for i, name in enumerate(layer_names):
        for j, b in enumerate(bits_ladder):
            with quantize_layer(name, b):
                v = float(evaluate())
            absolute[i, j] = v
            delta[i, j] = (baseline - v) if higher_is_better else (v - baseline)
        if progress:
            worst = np.nanmax(delta[i]) if np.isfinite(delta[i]).any() else float("nan")
            print(f"  [{i + 1:>2}/{n_l}] {name:<40s} worst Delta{metric} = {worst:+.4f}")

    return SensitivityResult(
        layer_names=layer_names,
        bits_ladder=bits_ladder,
        delta=delta,
        absolute=absolute,
        baseline=baseline,
        metric=metric,
        higher_is_better=higher_is_better,
        domain=domain,
        backbone=backbone,
        elapsed_s=time.time() - t0,
    )


def sensitivity_at_bits(result: SensitivityResult, bits: int) -> np.ndarray:
    return result.sensitivity_at(bits)


def normalise_sensitivity(s: np.ndarray, mode: str = "max") -> np.ndarray:
    """Normalize a sensitivity vector to [0, 1] for cross-backbone comparison.

    mode
    ----
    max  : divide by the maximum (keeps relative magnitudes, the usual choice)
    rank : map to a 0..1 quantile order (keeps only the ordering, equivalent to
           using the ranks directly)
    none : return unchanged
    """
    s = np.asarray(s, dtype=np.float64)
    if mode == "none":
        return s
    if mode == "rank":
        from .rank_corr import rankdata_average

        r = rankdata_average(s)
        if r.size <= 1:
            return np.zeros_like(r)
        return (r - 1) / (r.size - 1)
    if mode == "max":
        m = np.nanmax(np.abs(s))
        if not np.isfinite(m) or m < 1e-12:
            return np.zeros_like(s)
        return s / m
    raise ValueError(f"unknown normalization mode: {mode}")
