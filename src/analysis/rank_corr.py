"""秩相关性分析 —— 项目的核心量 ρ 在这里算。

为什么不能只报一个 Spearman ρ
------------------------------
本项目的层数只有十几层。在 n≈12 的样本上，Spearman ρ 的抽样方差**很大**：
即使两个域完全独立，|ρ| 也可能轻松超过 0.5。

所以这里额外提供：
  * ``permutation_test`` —— 给出 p 值（H0：两域敏感性排序独立）
  * ``bootstrap_ci``     —— 给出 ρ 的置信区间

**只报 ρ 不报显著性是不严谨的**，答辩时一定会被问。

参考：Spearman 秩相关的平均秩处理见 Kendall (1970)；置换检验见 Good (2005)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "rankdata_average",
    "pearson",
    "spearman",
    "kendall_tau_b",
    "permutation_test",
    "bootstrap_ci",
    "RankCorrelationResult",
    "analyse_pair",
]


# --------------------------------------------------------------------------
# 秩
# --------------------------------------------------------------------------
def rankdata_average(a: np.ndarray) -> np.ndarray:
    """平均秩（并列取平均）。等价于 scipy.stats.rankdata(method="average")。

    自己实现是为了让核心分析不依赖 scipy，并且逻辑可读、可测试。
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    n = a.size
    if n == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # 名次是 1-based
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _drop_nan(*arrs: np.ndarray) -> Tuple[np.ndarray, ...]:
    arrs = [np.asarray(a, dtype=np.float64).ravel() for a in arrs]
    n = arrs[0].size
    for a in arrs:
        if a.size != n:
            raise ValueError("所有输入长度必须一致")
    mask = np.ones(n, dtype=bool)
    for a in arrs:
        mask &= np.isfinite(a)
    return tuple(a[mask] for a in arrs)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _drop_nan(a, b)
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-15:
        return float("nan")
    return float((a * b).sum() / denom)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关（= 平均秩上的 Pearson）。"""
    a, b = _drop_nan(a, b)
    if a.size < 2:
        return float("nan")
    return pearson(rankdata_average(a), rankdata_average(b))


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall τ-b。层数少时比 Spearman 更稳健，可作为附加证据。"""
    a, b = _drop_nan(a, b)
    n = a.size
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                ties_a += 1
                ties_b += 1
            elif da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = concordant + discordant
    denom = np.sqrt((n0 + ties_a) * (n0 + ties_b))
    if denom < 1e-15:
        return float("nan")
    return float((concordant - discordant) / denom)


# --------------------------------------------------------------------------
# 显著性
# --------------------------------------------------------------------------
def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
    two_sided: bool = True,
) -> Tuple[float, float]:
    """置换检验。返回 ``(rho_observed, p_value)``。

    H0：两个域的敏感性排序相互独立。
    在 H0 下，打乱 b 的对应关系不改变 ρ 的分布，于是可以精确地重采样出零分布。
    """
    a, b = _drop_nan(a, b)
    rho = spearman(a, b)
    if not np.isfinite(rho) or a.size < 3:
        return rho, float("nan")

    rng = np.random.default_rng(seed)
    ra = rankdata_average(a)
    rb = rankdata_average(b)
    count = 0
    for _ in range(n_permutations):
        rb_perm = rng.permutation(rb)
        r = pearson(ra, rb_perm)
        if two_sided:
            if abs(r) >= abs(rho) - 1e-12:
                count += 1
        else:
            if r >= rho - 1e-12:
                count += 1
    # +1 修正，避免 p=0
    p = (count + 1) / (n_permutations + 1)
    return rho, float(p)


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """ρ 的 percentile bootstrap 置信区间。返回 ``(lo, hi)``。"""
    a, b = _drop_nan(a, b)
    n = a.size
    if n < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = spearman(a[idx], b[idx])
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


# --------------------------------------------------------------------------
# 结果容器
# --------------------------------------------------------------------------
@dataclass
class RankCorrelationResult:
    """一对域之间的秩稳定性结果。这是报告里最重要的一个数据类。"""

    n_layers: int
    rho: float
    p_value: float
    ci_low: float
    ci_high: float
    tau_b: float
    rho_pearson_raw: float  # 未转秩的 Pearson，作为对照（会被量纲/离群值影响）
    seed_used: int

    @property
    def significant(self) -> bool:
        """是否在 0.05 水平显著。显式返回 Python bool，避免 numpy 类型污染 JSON。"""
        p = self.p_value
        if not np.isfinite(p):
            return False
        return bool(p < 0.05)

    def verdict(self, alpha: float = 0.05) -> str:
        """把数字翻译成结论。这是写报告时直接可用的一句话。"""
        if not np.isfinite(self.rho):
            return "样本不足，无法判断"
        if np.isfinite(self.p_value) and self.p_value >= alpha:
            return (
                f"ρ={self.rho:+.3f} 但不显著（p={self.p_value:.3f}），"
                "不能拒绝'两域敏感性排序独立'的原假设"
            )
        if self.rho >= 0.8:
            return f"ρ={self.rho:+.3f}（p={self.p_value:.3g}）：排序高度一致 → 敏感性近似为架构内在属性"
        if self.rho >= 0.5:
            return f"ρ={self.rho:+.3f}（p={self.p_value:.3g}）：排序中度一致 → 部分可迁移"
        if self.rho >= 0:
            return f"ρ={self.rho:+.3f}（p={self.p_value:.3g}）：排序弱一致 → 位宽分配基本不可跨域复用"
        return f"ρ={self.rho:+.3f}（p={self.p_value:.3g}）：排序反转 → 跨域复用会主动损害性能"

    def as_row(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "rho": round(self.rho, 4),
            "p_value": round(self.p_value, 6),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "tau_b": round(self.tau_b, 4),
            "significant": self.significant,
            "verdict": self.verdict(),
        }


def analyse_pair(
    sens_a: np.ndarray,
    sens_b: np.ndarray,
    n_permutations: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    n_boot: int = 5000,
) -> RankCorrelationResult:
    """给定两个域的逐层敏感性向量，产出完整的秩稳定性分析。"""
    a, b = _drop_nan(sens_a, sens_b)
    rho, p = permutation_test(a, b, n_permutations=n_permutations, seed=seed)
    lo, hi = bootstrap_ci(a, b, n_boot=n_boot, alpha=alpha, seed=seed)
    return RankCorrelationResult(
        n_layers=int(a.size),
        rho=float(rho),
        p_value=float(p),
        ci_low=float(lo),
        ci_high=float(hi),
        tau_b=kendall_tau_b(a, b),
        rho_pearson_raw=pearson(a, b),
        seed_used=int(seed),
    )
