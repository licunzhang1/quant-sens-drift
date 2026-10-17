"""The equal-reconstruction-error arm -- upgrading "the ranking changes" to
"no domain-independent optimal bit allocation exists".

Problem setting
---------------
Assign each layer l a bit width ``b_l  in  bit-width ladder`` and, subject to a
**fixed total quantization error budget**, minimise the predicted damage

    D(b) = Sum_l  s_l * eps_l(b_l)

where ``s_l`` is the sensitivity of layer l and ``eps_l(b)`` is the reconstruction
error (NMSE) of layer l at b bits.

The three arms
--------------
1. **uniform**  every layer at the same bit width (sensitivity ignored)
2. **agnostic** allocate with the sensitivity of **source domain A**  <- this is
   what is actually possible in practice (you only know the training domain)
3. **oracle**   allocate with the sensitivity of **target domain B** <- not
   implementable, it is the upper bound

Two key differences
-------------------
* ``agnostic - uniform``: does sensitivity-guided allocation help at all?
* ``oracle - agnostic``: **the cost of "not knowing the deployment domain"** <-
  this is the core quantity of the project

If the latter cost is not negligible, we get a clean counterexample:

    **No domain-independent optimal bit allocation exists.**

The implementation is greedy (marginal gain / marginal budget); it is
near-optimal for this kind of separable concave problem and fully reproducible.
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
    """``D = Sum_l s_l * eps_l(b_l)``."""
    bits = np.asarray(bits, dtype=int)
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    err_curves = np.asarray(err_curves, dtype=np.float64)
    ladder = list(bits_ladder)
    idx = np.array([ladder.index(int(b)) for b in bits], dtype=int)
    eps = err_curves[np.arange(len(bits)), idx]
    return float(np.sum(sensitivity * eps))


@dataclass
class Allocation:
    """The result of one bit-width allocation."""

    bits: np.ndarray                 # (n_layers,) bit width per layer
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
# budget
# --------------------------------------------------------------------------
def _resolve_costs(n_layers: int, layer_costs: Optional[Sequence[float]]) -> np.ndarray:
    if layer_costs is None:
        return np.ones(n_layers, dtype=np.float64)
    c = np.asarray(layer_costs, dtype=np.float64).ravel()
    if c.size != n_layers:
        raise ValueError(f"layer_costs has length {c.size}; expected {n_layers} layers")
    c = np.maximum(c, 1e-12)
    return c


def _total_budget(costs: np.ndarray, bits_ladder: Sequence[int], budget_ratio: float) -> float:
    max_bits = float(max(bits_ladder))
    return float(budget_ratio * np.sum(costs) * max_bits)


# --------------------------------------------------------------------------
# allocation strategies
# --------------------------------------------------------------------------
def uniform_allocate(
    n_layers: int,
    bits_ladder: Sequence[int],
    budget_ratio: float = 0.8,
    layer_costs: Optional[Sequence[float]] = None,
    sensitivity: Optional[np.ndarray] = None,
    err_curves: Optional[np.ndarray] = None,
) -> Allocation:
    """Give every layer the same bit width: the largest rung the budget allows."""
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
    """Greedy bit-width allocation.

    Each step raises by one rung the layer with the largest "marginal damage
    reduction / marginal budget".

    Parameters
    ----------
    sensitivity : (n_layers,)  s_l, larger means more sensitive
    err_curves  : (n_layers, n_bits)  eps_l(b), ordered as bits_ladder, decreasing
                  monotonically with bit width
    budget_ratio: total budget = ratio x Sum_l costs_l x max(bits_ladder)
    layer_costs : budget weight per layer (parameter count by default; None means
                  all ones)
    min_bits    : starting bit width, defaults to the smallest ladder entry
    """
    s = np.asarray(sensitivity, dtype=np.float64).ravel()
    E = np.asarray(err_curves, dtype=np.float64)
    ladder = [int(b) for b in bits_ladder]
    n_l, n_b = E.shape
    if s.size != n_l:
        raise ValueError(f"sensitivity has length {s.size}; expected {n_l} rows")
    if n_b != len(ladder):
        raise ValueError("number of err_curves columns must equal bit-width ladder length")

    # the sensitivity can be negative (quantization even "helps", pure noise).
    # Clip it at 0, otherwise the greedy search would lower the bit width of
    # those layers in order to "maximise the gain", which makes no sense.
    s = np.clip(np.nan_to_num(s, nan=0.0), 0.0, None)

    costs = _resolve_costs(n_l, layer_costs)
    budget = _total_budget(costs, ladder, budget_ratio)

    start_idx = 0
    if min_bits is not None:
        if int(min_bits) not in ladder:
            raise ValueError(f"min_bits={min_bits} is not in the bit-width ladder")
        start_idx = ladder.index(int(min_bits))

    idx = np.full(n_l, start_idx, dtype=int)
    used = float(np.sum(costs * np.array([ladder[i] for i in idx], dtype=np.float64)))

    # if even the minimum bit width exceeds the budget, return the minimal
    # configuration right away (and report the budget honestly)
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
# three-arm comparison
# --------------------------------------------------------------------------
@dataclass
class EqualErrorArmResult:
    """The complete result of the equal-reconstruction-error arm."""

    layer_names: List[str]
    bits_ladder: List[int]
    budget_ratio: float
    budget_total: float
    uniform: Allocation
    agnostic: Allocation       # uses the sensitivity of source domain A
    oracle: Allocation         # uses the sensitivity of target domain B (upper bound)
    source_domain: str = ""
    target_domain: str = ""

    # the two key differences (both evaluate the predicted damage with the
    # **target-domain** sensitivity)
    @property
    def gain_from_sensitivity(self) -> float:
        """uniform - agnostic: gain from sensitivity-guided allocation (>0 means it helps)."""
        return float(self.uniform.predicted_damage - self.agnostic.predicted_damage)

    @property
    def cost_of_domain_ignorance(self) -> float:
        """agnostic - oracle: **the cost of not knowing the deployment domain** (>=0)."""
        return float(self.agnostic.predicted_damage - self.oracle.predicted_damage)

    @property
    def relative_cost_of_ignorance(self) -> float:
        """Relative cost: the extra damage paid, relative to the oracle's damage."""
        denom = abs(self.oracle.predicted_damage)
        if denom < 1e-12:
            return float("nan")
        return self.cost_of_domain_ignorance / denom

    @property
    def ranking_flipped(self) -> bool:
        """Whether the two allocations differ (whether the ranking changed the decision)."""
        return not np.array_equal(np.asarray(self.agnostic.bits), np.asarray(self.oracle.bits))

    def verdict(self) -> str:
        d = self.cost_of_domain_ignorance
        if not np.isfinite(d) or abs(d) < 1e-9:
            return ("domain-independent and domain-aware allocation are "
                    "equivalent -- no cost of 'not knowing the deployment "
                    "domain' was found")
        if not self.ranking_flipped:
            return (
                "both allocations give the same bit widths -- the ranking "
                "changed but not the decision, so the shift in the sensitivity "
                "ranking has no practical effect under this budget"
            )
        return (
            f"allocating by source-domain sensitivity pays {d:.4g} more "
            f"predicted damage than allocating by target-domain sensitivity "
            f"(relative {self.relative_cost_of_ignorance:+.1%}) -> "
            "**no domain-independent optimal bit allocation exists**"
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
    """Run the complete three-arm experiment.

    Note that the predicted damage is always evaluated with the **target-domain**
    sensitivity, because the question is "how good is this allocation on the
    deployment domain B".
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

    # recompute the damage with the target-domain sensitivity throughout, so
    # that the three arms are comparable
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
