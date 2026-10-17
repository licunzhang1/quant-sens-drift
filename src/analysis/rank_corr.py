"""Rank-correlation analysis -- the core quantity rho of the project is computed
here.

Why reporting a single Spearman rho is not enough
-----------------------------------------------
This project has only a dozen-odd layers. On a sample of n~=12 the sampling
variance of Spearman rho is **large**: even if the two domains are completely
independent, |rho| can easily exceed 0.5.

So this module additionally provides:
  * ``permutation_test`` -- gives a p-value (H0: the two domains' sensitivity
    rankings are independent)
  * ``bootstrap_ci``     -- gives a confidence interval for rho

**Reporting rho without a significance test is not rigorous**, and it will
certainly be questioned in a defense.

Reference: average-rank handling for the Spearman rank correlation follows
Kendall (1970); the permutation test follows Good (2005).
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
# ranks
# --------------------------------------------------------------------------
def rankdata_average(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank).

    Equivalent to scipy.stats.rankdata(method="average"). It is implemented here
    so that the core analysis does not depend on scipy, and so that the logic
    stays readable and testable.
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
            avg = (i + j + 2) / 2.0  # the ranks are 1-based
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _drop_nan(*arrs: np.ndarray) -> Tuple[np.ndarray, ...]:
    arrs = [np.asarray(a, dtype=np.float64).ravel() for a in arrs]
    n = arrs[0].size
    for a in arrs:
        if a.size != n:
            raise ValueError("all inputs must have the same length")
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
    """Spearman rank correlation (= Pearson on the average ranks)."""
    a, b = _drop_nan(a, b)
    if a.size < 2:
        return float("nan")
    return pearson(rankdata_average(a), rankdata_average(b))


def kendall_tau_b(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b. More robust than Spearman with few layers; useful as
    supporting evidence."""
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
# significance
# --------------------------------------------------------------------------
def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
    two_sided: bool = True,
) -> Tuple[float, float]:
    """Permutation test. Returns ``(rho_observed, p_value)``.

    H0: the sensitivity rankings of the two domains are independent.
    Under H0, shuffling the pairing of b does not change the distribution of rho,
    so the null distribution can be resampled exactly.
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
    # +1 correction so that p never reaches 0
    p = (count + 1) / (n_permutations + 1)
    return rho, float(p)


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for rho. Returns ``(lo, hi)``."""
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
# result container
# --------------------------------------------------------------------------
@dataclass
class RankCorrelationResult:
    """Rank-stability result for a pair of domains.

    This is the single most important data class in the report.
    """

    n_layers: int
    rho: float
    p_value: float
    ci_low: float
    ci_high: float
    tau_b: float
    rho_pearson_raw: float  # Pearson without ranking, as a control (moves with units/outliers)
    seed_used: int

    @property
    def significant(self) -> bool:
        """Whether the result is significant at the 0.05 level.

        Returns a Python bool explicitly so that numpy types do not leak into
        the JSON output.
        """
        p = self.p_value
        if not np.isfinite(p):
            return False
        return bool(p < 0.05)

    def verdict(self, alpha: float = 0.05) -> str:
        """Turn the numbers into a conclusion.

        A sentence that can be used directly when writing the report.
        """
        if not np.isfinite(self.rho):
            return "not enough samples to decide"
        if np.isfinite(self.p_value) and self.p_value >= alpha:
            return (
                f"rho={self.rho:+.3f} but not significant (p={self.p_value:.3f}); "
                "cannot reject the null hypothesis that the two domains' "
                "sensitivity rankings are independent"
            )
        if self.rho >= 0.8:
            return (f"rho={self.rho:+.3f} (p={self.p_value:.3g}): rankings highly "
                    "consistent -> sensitivity is close to an intrinsic property "
                    "of the architecture")
        if self.rho >= 0.5:
            return (f"rho={self.rho:+.3f} (p={self.p_value:.3g}): rankings "
                    "moderately consistent -> partly transferable")
        if self.rho >= 0:
            return (f"rho={self.rho:+.3f} (p={self.p_value:.3g}): rankings weakly "
                    "consistent -> bit allocation is basically not reusable "
                    "across domains")
        return (f"rho={self.rho:+.3f} (p={self.p_value:.3g}): rankings reversed -> "
                "reusing them across domains actively hurts performance")

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
    """Given the layer-wise sensitivity vectors of two domains, produce the full
    rank-stability analysis."""
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
