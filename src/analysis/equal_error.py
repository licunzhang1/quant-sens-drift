"""等重构误差臂 —— 把"排序会变"升级为"不存在域无关的最优比特分配"。

问题设定
--------
给每个层 ℓ 分配位宽 ``b_ℓ ∈ 位宽阶梯``，在**总量化误差预算固定**的前提下，
最小化预测损伤

    D(b) = Σ_ℓ  s_ℓ · ε_ℓ(b_ℓ)

其中 ``s_ℓ`` 是第 ℓ 层的敏感性，``ε_ℓ(b)`` 是第 ℓ 层在 b bit 下的重构误差（NMSE）。

三条臂
------
1. **uniform**  所有层同位宽（不看敏感性）
2. **agnostic** 用**源域 A** 的敏感性分配  ← 这就是现实中你能做的（你只知道训练域）
3. **oracle**   用**目标域 B** 的敏感性分配 ← 不可实现，是上界

两个关键差值
------------
* ``agnostic − uniform``：敏感性引导的分配到底有没有用？
* ``oracle − agnostic``：**"不知道部署域"的代价** ← 这是本项目的核心量

若后者的代价不可忽略，就得到干净的反例：

    **不存在一个域无关的最优比特分配。**

实现是贪心（边际收益 / 边际预算），对这类可分凹问题近似最优，且完全可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Allocation",
    "greedy_allocate",
    "uniform_allocate",
    "EqualErrorArmResult",
    "run_equal_error_arm",
    "predicted_damage",
]


def predicted_damage(
    bits: np.ndarray,
    sensitivity: np.ndarray,
    err_curves: np.ndarray,
    bits_ladder: Sequence[int],
) -> float:
    """``D = Σ_ℓ s_ℓ · ε_ℓ(b_ℓ)``。"""
    bits = np.asarray(bits, dtype=int)
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    err_curves = np.asarray(err_curves, dtype=np.float64)
    ladder = list(bits_ladder)
    idx = np.array([ladder.index(int(b)) for b in bits], dtype=int)
    eps = err_curves[np.arange(len(bits)), idx]
    return float(np.sum(sensitivity * eps))


@dataclass
class Allocation:
    """一次位宽分配的结果。"""

    bits: np.ndarray                 # (n_layers,) 每层位宽
    predicted_damage: float
    budget_used: float
    budget_total: float
    strategy: str
    layer_costs: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def budget_ratio_used(self) -> float:
        return self.budget_used / self.budget_total if self.budget_total > 0 else float("nan")

    def summary(self) -> Dict[str, object]:
        bits = np.asarray(self.bits)
        return {
            "strategy": self.strategy,
            "bits": bits.tolist(),
            "mean_bits": float(bits.mean()),
            "min_bits": int(bits.min()),
            "max_bits": int(bits.max()),
            "predicted_damage": round(self.predicted_damage, 6),
            "budget_used": round(self.budget_used, 3),
            "budget_total": round(self.budget_total, 3),
        }


# --------------------------------------------------------------------------
# 预算
# --------------------------------------------------------------------------
def _resolve_costs(n_layers: int, layer_costs: Optional[Sequence[float]]) -> np.ndarray:
    if layer_costs is None:
        return np.ones(n_layers, dtype=np.float64)
    c = np.asarray(layer_costs, dtype=np.float64).ravel()
    if c.size != n_layers:
        raise ValueError(f"layer_costs 长度 {c.size} 与层数 {n_layers} 不一致")
    c = np.maximum(c, 1e-12)
    return c


def _total_budget(costs: np.ndarray, bits_ladder: Sequence[int], budget_ratio: float) -> float:
    max_bits = float(max(bits_ladder))
    return float(budget_ratio * np.sum(costs) * max_bits)


# --------------------------------------------------------------------------
# 分配策略
# --------------------------------------------------------------------------
def uniform_allocate(
    n_layers: int,
    bits_ladder: Sequence[int],
    budget_ratio: float = 0.8,
    layer_costs: Optional[Sequence[float]] = None,
    sensitivity: Optional[np.ndarray] = None,
    err_curves: Optional[np.ndarray] = None,
) -> Allocation:
    """所有层用同一个位宽：取预算允许的最大那一档。"""
    ladder = list(bits_ladder)
    costs = _resolve_costs(n_layers, layer_costs)
    budget = _total_budget(costs, ladder, budget_ratio)

    chosen = ladder[0]
    for b in ladder:
        if float(np.sum(costs) * b) <= budget + 1e-9:
            chosen = b
    bits = np.full(n_layers, chosen, dtype=int)

    dmg = float("nan")
    if sensitivity is not None and err_curves is not None:
        dmg = predicted_damage(bits, sensitivity, err_curves, ladder)
    return Allocation(bits, dmg, float(np.sum(costs) * chosen), budget, "uniform", costs)


def greedy_allocate(
    sensitivity: np.ndarray,
    err_curves: np.ndarray,
    bits_ladder: Sequence[int],
    budget_ratio: float = 0.8,
    layer_costs: Optional[Sequence[float]] = None,
    min_bits: Optional[int] = None,
    strategy: str = "greedy",
) -> Allocation:
    """贪心位宽分配：每步选"边际损伤下降 / 边际预算"最大的那一层升一档。

    Parameters
    ----------
    sensitivity : (n_layers,)   s_ℓ，越大越敏感
    err_curves  : (n_layers, n_bits)  ε_ℓ(b)，按 bits_ladder 排列，随位宽单调下降
    budget_ratio: 总预算 = ratio × Σ_ℓ costs_ℓ × max(bits_ladder)
    layer_costs : 每层的预算权重（默认参数量；None 表示全 1）
    min_bits    : 起始位宽，默认取 ladder 最小值
    """
    s = np.asarray(sensitivity, dtype=np.float64).ravel()
    E = np.asarray(err_curves, dtype=np.float64)
    ladder = [int(b) for b in bits_ladder]
    n_l, n_b = E.shape
    if s.size != n_l:
        raise ValueError(f"sensitivity 长度 {s.size} 与 err_curves 行数 {n_l} 不一致")
    if n_b != len(ladder):
        raise ValueError("err_curves 列数必须等于位宽阶梯长度")

    # 敏感性可能是负的（量化反而"变好"，纯噪声）。截断到 0，
    # 否则贪心会为了"最大化收益"去降低这些层的位宽，语义上说不通。
    s = np.clip(np.nan_to_num(s, nan=0.0), 0.0, None)

    costs = _resolve_costs(n_l, layer_costs)
    budget = _total_budget(costs, ladder, budget_ratio)

    start_idx = 0
    if min_bits is not None:
        if int(min_bits) not in ladder:
            raise ValueError(f"min_bits={min_bits} 不在位宽阶梯中")
        start_idx = ladder.index(int(min_bits))

    idx = np.full(n_l, start_idx, dtype=int)
    used = float(np.sum(costs * np.array([ladder[i] for i in idx], dtype=np.float64)))

    # 若最小位宽都超预算，直接返回最小配置（并在 budget 上如实反映）
    while True:
        best_l, best_gain, best_score = -1, 0.0, -np.inf
        for l in range(n_l):
            if idx[l] >= n_b - 1:
                continue
            b_cur = ladder[idx[l]]
            b_next = ladder[idx[l] + 1]
            d_cost = costs[l] * (b_next - b_cur)
            if used + d_cost > budget + 1e-9:
                continue
            d_gain = s[l] * (E[l, idx[l]] - E[l, idx[l] + 1])
            if not np.isfinite(d_gain) or d_gain <= 0:
                continue
            score = d_gain / d_cost
            if score > best_score:
                best_l, best_gain, best_score = l, d_gain, score
        if best_l < 0:
            break
        b_cur = ladder[idx[best_l]]
        b_next = ladder[idx[best_l] + 1]
        idx[best_l] += 1
        used += costs[best_l] * (b_next - b_cur)

    bits = np.array([ladder[i] for i in idx], dtype=int)
    dmg = predicted_damage(bits, s, E, ladder)
    return Allocation(bits, dmg, used, budget, strategy, costs)


# --------------------------------------------------------------------------
# 三臂对比
# --------------------------------------------------------------------------
@dataclass
class EqualErrorArmResult:
    """等重构误差臂的完整结果。"""

    layer_names: List[str]
    bits_ladder: List[int]
    budget_ratio: float
    budget_total: float
    uniform: Allocation
    agnostic: Allocation       # 用源域 A 的敏感性
    oracle: Allocation         # 用目标域 B 的敏感性（上界）
    source_domain: str = ""
    target_domain: str = ""

    # 两个关键差值（都用**目标域**敏感性算预测损伤）
    @property
    def gain_from_sensitivity(self) -> float:
        """uniform − agnostic：敏感性引导分配带来的收益（>0 说明有用）。"""
        return float(self.uniform.predicted_damage - self.agnostic.predicted_damage)

    @property
    def cost_of_domain_ignorance(self) -> float:
        """agnostic − oracle：**不知道部署域的代价**（>=0）。"""
        return float(self.agnostic.predicted_damage - self.oracle.predicted_damage)

    @property
    def relative_cost_of_ignorance(self) -> float:
        """相对代价：相对于 oracle 的损伤，多付出的比例。"""
        denom = abs(self.oracle.predicted_damage)
        if denom < 1e-12:
            return float("nan")
        return self.cost_of_domain_ignorance / denom

    @property
    def ranking_flipped(self) -> bool:
        """两份分配是否不同（排序是否真的改变了决策）。"""
        return not np.array_equal(np.asarray(self.agnostic.bits), np.asarray(self.oracle.bits))

    def verdict(self) -> str:
        d = self.cost_of_domain_ignorance
        if not np.isfinite(d) or abs(d) < 1e-9:
            return "域无关分配与域相关分配等价 —— 未发现'不知道部署域'的代价"
        if not self.ranking_flipped:
            return (
                "两份分配位宽相同 —— 排序虽变但未改变决策，"
                "说明敏感性排序的变化在本次预算下不具实际影响"
            )
        return (
            f"按源域敏感性分配比按目标域分配多付出 {d:.4g} 的预测损伤"
            f"（相对 {self.relative_cost_of_ignorance:+.1%}）→ "
            "**不存在域无关的最优比特分配**"
        )

    def to_rows(self) -> List[Dict[str, object]]:
        rows = []
        for tag, alloc in (
            ("uniform", self.uniform),
            ("agnostic_source", self.agnostic),
            ("oracle_target", self.oracle),
        ):
            row = alloc.summary()
            row["arm"] = tag
            row["source_domain"] = self.source_domain
            row["target_domain"] = self.target_domain
            rows.append(row)
        return rows


def run_equal_error_arm(
    sens_source: np.ndarray,
    sens_target: np.ndarray,
    err_curves: np.ndarray,
    bits_ladder: Sequence[int] = (2, 3, 4, 6, 8),
    budget_ratio: float = 0.8,
    layer_costs: Optional[Sequence[float]] = None,
    layer_names: Optional[Sequence[str]] = None,
    source_domain: str = "A",
    target_domain: str = "B",
) -> EqualErrorArmResult:
    """跑完整的三臂实验。

    注意：预测损伤**统一用目标域敏感性**评估，因为我们要回答的是
    "在部署域 B 上，这个分配好不好"。
    """
    ladder = [int(b) for b in bits_ladder]
    n_l = np.asarray(err_curves).shape[0]
    names = list(layer_names) if layer_names is not None else [f"layer{i}" for i in range(n_l)]

    costs = _resolve_costs(n_l, layer_costs)
    budget = _total_budget(costs, ladder, budget_ratio)

    uni = uniform_allocate(n_l, ladder, budget_ratio, layer_costs, sens_target, err_curves)
    agn = greedy_allocate(sens_source, err_curves, ladder, budget_ratio, layer_costs,
                          strategy="agnostic_source")
    orc = greedy_allocate(sens_target, err_curves, ladder, budget_ratio, layer_costs,
                          strategy="oracle_target")

    # 统一用目标域敏感性重算损伤，保证三臂可比
    for alloc in (uni, agn, orc):
        alloc.predicted_damage = predicted_damage(alloc.bits, sens_target, err_curves, ladder)

    return EqualErrorArmResult(
        layer_names=names,
        bits_ladder=ladder,
        budget_ratio=float(budget_ratio),
        budget_total=budget,
        uniform=uni,
        agnostic=agn,
        oracle=orc,
        source_domain=source_domain,
        target_domain=target_domain,
    )
