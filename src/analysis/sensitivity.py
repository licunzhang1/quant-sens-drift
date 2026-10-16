"""逐层量化敏感性矩阵。

核心量
------
``Δ_l(b) = M(量化第 l 层到 b bit) - M(全浮点)``

其中 M 是评价指标（准确率或损失）。

设计上是**框架无关**的：只依赖两个回调
  * ``evaluate()``             -> float，全浮点下的指标
  * ``quantize_layer(name, b)`` -> context manager，临时把第 l 层量化

这样 numpy 冒烟测试和 torch 真实训练可以共用同一套逻辑。
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
    """逐层敏感性矩阵及其元数据。"""

    layer_names: List[str]
    bits_ladder: List[int]
    delta: np.ndarray          # (n_layers, n_bits)  Δmetric，已按"越大越糟"定向
    absolute: np.ndarray       # (n_layers, n_bits)  量化后的绝对指标
    baseline: float            # 全浮点指标
    metric: str
    higher_is_better: bool
    domain: str = ""
    backbone: str = ""
    elapsed_s: float = 0.0
    extra: Dict[str, object] = field(default_factory=dict)

    # ---------------------------------------------------------------- 便捷
    @property
    def n_layers(self) -> int:
        return len(self.layer_names)

    def sensitivity_at(self, bits: int) -> np.ndarray:
        """取某个位宽下的敏感性向量（秩分析通常用参考位宽）。"""
        if bits not in self.bits_ladder:
            raise KeyError(f"{bits} 不在位宽阶梯 {self.bits_ladder} 中")
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
    """主入口：算完整的逐层敏感性矩阵。

    Parameters
    ----------
    layer_names : 要分析的层名（顺序即输出行顺序）
    bits_ladder : 位宽阶梯，如 (2, 3, 4, 6, 8)
    evaluate : 无参函数，返回当前模型在目标域上的指标
    quantize_layer : ``(layer_name, bits) -> context manager``
    higher_is_better : True（准确率）时 Δ 定义为"下降量"；False（损失）时定义为"上升量"。
        两种情况都保证 **Δ 越大 = 越糟**，可以直接比较与排序。

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
            print(f"  [{i + 1:>2}/{n_l}] {name:<40s} worst Δ{metric} = {worst:+.4f}")

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
    """把敏感性向量归一化到 [0,1]，方便跨骨干比较。

    mode
    ----
    max  : 除以最大值（保持相对大小，最常用）
    rank : 转成 0..1 的分位序（只保留顺序信息，等价于直接用秩）
    none : 原样返回
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
    raise ValueError(f"未知的归一化模式: {mode}")
