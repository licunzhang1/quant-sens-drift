from .equal_error import (
    Allocation,
    EqualErrorArmResult,
    greedy_allocate,
    predicted_damage,
    run_equal_error_arm,
    uniform_allocate,
)
from .proxies import correlate_with_sensitivity, proxy_matrix, proxy_vector
from .rank_corr import (
    RankCorrelationResult,
    analyse_pair,
    bootstrap_ci,
    kendall_tau_b,
    pearson,
    permutation_test,
    rankdata_average,
    spearman,
)
from .sensitivity import SensitivityResult, compute_sensitivity_matrix, normalise_sensitivity

__all__ = [
    "Allocation",
    "EqualErrorArmResult",
    "greedy_allocate",
    "predicted_damage",
    "run_equal_error_arm",
    "uniform_allocate",
    "correlate_with_sensitivity",
    "proxy_matrix",
    "proxy_vector",
    "RankCorrelationResult",
    "analyse_pair",
    "bootstrap_ci",
    "kendall_tau_b",
    "pearson",
    "permutation_test",
    "rankdata_average",
    "spearman",
    "SensitivityResult",
    "compute_sensitivity_matrix",
    "normalise_sensitivity",
]
