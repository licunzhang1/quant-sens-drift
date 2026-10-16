"""报告用的四张核心图。

图 1  逐层敏感性热图（层 × 位宽，两个域并排）
图 2  敏感性排序散点 + ρ + p 值（核心量）
图 3  等重构误差三臂对比（最关键的一张）
图 4  代理量 vs 敏感性散点

只依赖 numpy + matplotlib，不需要 torch。
**图内文字用英文**：避免 CJK 字体缺失导致的方框乱码；
中文说明放在 README / 报告正文里。
"""

from __future__ import annotations

import pathlib
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "plot_sensitivity_heatmap",
    "plot_rank_scatter",
    "plot_equal_error_arms",
    "plot_proxy_scatter",
    "plot_domain_gap",
]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: str | pathlib.Path, dpi: int = 200) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def plot_sensitivity_heatmap(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    layer_names: Sequence[str],
    bits: Sequence[int],
    path: str | pathlib.Path,
    name_a: str = "Domain A (source)",
    name_b: str = "Domain B (target)",
) -> pathlib.Path:
    """图 1：逐层敏感性热图。同一个色标，方便直接比较两个域。"""
    plt = _plt()
    a = np.asarray(delta_a, dtype=float)
    b = np.asarray(delta_b, dtype=float)
    vmax = np.nanmax(np.abs(np.concatenate([a.ravel(), b.ravel()]))) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, 0.42 * len(layer_names))),
                             sharey=True)
    for ax, mat, title in ((axes[0], a, name_a), (axes[1], b, name_b)):
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(bits)))
        ax.set_xticklabels([f"{int(x)}b" for x in bits])
        ax.set_xlabel("bit-width")
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, label=r"$\Delta$ metric (higher = worse)")
    fig.suptitle("Layer-wise quantization sensitivity", fontsize=13)
    return _save(fig, path)


def plot_rank_scatter(
    sens_a: np.ndarray,
    sens_b: np.ndarray,
    layer_names: Sequence[str],
    rho: float,
    p_value: float,
    path: str | pathlib.Path,
    reference_bits: Optional[int] = None,
    annotate: bool = True,
) -> pathlib.Path:
    """图 2：两域敏感性的排序散点 —— 核心量 ρ 的可视化。"""
    plt = _plt()
    a = np.asarray(sens_a, dtype=float)
    b = np.asarray(sens_b, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    ax.scatter(a, b, s=42, alpha=0.85, edgecolor="k", linewidth=0.5)
    if annotate:
        for x, y, nm in zip(a, b, layer_names):
            ax.annotate(str(nm)[:14], (x, y), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("sensitivity in source domain" +
                  (f" ({reference_bits}b)" if reference_bits else ""))
    ax.set_ylabel("sensitivity in target domain")
    ax.set_title(f"Layer sensitivity: $\\rho$={rho:+.3f}, p={p_value:.3g}", fontsize=11)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ra = _ranks(a)
    rb = _ranks(b)
    ax.scatter(ra, rb, s=42, alpha=0.85, edgecolor="k", linewidth=0.5)
    lim = [0.5, len(a) + 0.5]
    ax.plot(lim, lim, color="tab:red", lw=1, ls="--", label="perfect agreement")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("rank (source)")
    ax.set_ylabel("rank (target)")
    ax.set_title("Rank space (what $\\rho$ actually measures)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    return _save(fig, path)


def _ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    return r


def plot_equal_error_arms(
    arms: Dict[str, np.ndarray],
    damages: Dict[str, float],
    bits_ladder: Sequence[int],
    layer_names: Sequence[str],
    path: str | pathlib.Path,
    budget_ratio: Optional[float] = None,
) -> pathlib.Path:
    """图 3：等重构误差三臂对比 —— **最关键的一张图**。"""
    plt = _plt()
    order = ["uniform", "agnostic", "oracle"]
    labels = {
        "uniform": "uniform\n(no sensitivity)",
        "agnostic": "agnostic\n(source-domain sens.)",
        "oracle": "oracle\n(target-domain sens.)",
    }
    order = [o for o in order if o in arms] or list(arms.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    n_l = len(layer_names)
    width = 0.8 / max(len(order), 1)
    xs = np.arange(n_l)
    for k, key in enumerate(order):
        ax.bar(xs + k * width - 0.4 + width / 2, np.asarray(arms[key]),
               width=width, label=labels.get(key, key))
    ax.set_xticks(xs)
    ax.set_xticklabels(layer_names, rotation=90, fontsize=7)
    ax.set_ylabel("allocated bit-width")
    ax.set_title("Per-layer bit allocation under equal error budget", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    vals = [float(damages[k]) for k in order]
    bars = ax.bar([labels.get(k, k) for k in order], vals,
                  color=["tab:gray", "tab:orange", "tab:green"][: len(order)])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.4g}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_ylabel("predicted damage (target domain)")
    ttl = "Predicted damage"
    if budget_ratio is not None:
        ttl += f"  (budget = {budget_ratio:g})"
    ax.set_title(ttl, fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    if "agnostic" in damages and "oracle" in damages:
        gap = float(damages["agnostic"]) - float(damages["oracle"])
        ax.annotate(f"cost of domain ignorance\n= {gap:+.4g}",
                    xy=(0.5, 0.02), xycoords="axes fraction",
                    ha="center", fontsize=9,
                    bbox=dict(boxstyle="round", fc="lightyellow", ec="gray"))
    return _save(fig, path)


def plot_proxy_scatter(
    proxies: Dict[str, np.ndarray],
    sensitivity: np.ndarray,
    correlations: Sequence[Dict],
    path: str | pathlib.Path,
    top_k: int = 6,
) -> pathlib.Path:
    """图 4：代理量 vs 敏感性。"""
    plt = _plt()
    s = np.asarray(sensitivity, dtype=float)
    chosen = [c["proxy"] for c in correlations[:top_k] if c["proxy"] in proxies]
    if not chosen:
        chosen = list(proxies.keys())[:top_k]

    ncol = min(3, len(chosen))
    nrow = int(np.ceil(len(chosen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)

    lookup = {c["proxy"]: c for c in correlations}
    for i, pname in enumerate(chosen):
        ax = axes[i // ncol][i % ncol]
        ax.scatter(np.asarray(proxies[pname], dtype=float), s,
                   s=40, alpha=0.85, edgecolor="k", linewidth=0.5)
        c = lookup.get(pname, {})
        ax.set_title(f"{pname}\n$\\rho$={c.get('rho', float('nan')):+.3f}, "
                     f"p={c.get('p_value', float('nan')):.3g}", fontsize=9)
        ax.set_xlabel(pname)
        ax.set_ylabel("sensitivity (target)")
        ax.grid(alpha=0.25)
    for j in range(len(chosen), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Can cheap proxies predict quantization sensitivity?", fontsize=12)
    return _save(fig, path)


def plot_domain_gap(
    values: Dict[str, float],
    path: str | pathlib.Path,
    metric: str = "accuracy",
) -> pathlib.Path:
    """附加图：各域指标柱状图（展示域间隙）。"""
    plt = _plt()
    names = list(values.keys())
    vals = [float(values[k]) for k in names]
    fig, ax = plt.subplots(figsize=(max(4.5, 1.4 * len(names)), 3.8))
    bars = ax.bar(names, vals, color="tab:blue", alpha=0.85)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across domains", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, path)
