"""跨随机种子聚合：把点估计变成"均值 ± 标准差"。

为什么必须做
------------
层数只有 10~16 时，ρ 的抽样方差很大：单一种子下 10 个组合里有 7 个 p > 0.05。
但"单次不显著"不等于"没有效应" —— 多个独立种子下如果方向一致，
用 Fisher 合并 p 值可以把统计功效找回来。

本脚本输出三样东西：
  1. 每个 (骨干, 偏移) 的 ρ 与域无知代价的**均值 ± 标准差**；
  2. **Fisher 合并 p 值**（跨种子），以及显著种子数；
  3. 结论的**方向一致性**统计 —— 这是比单个 ρ 更稳健的量。

用法
----
    python scripts/aggregate_seeds.py
    python scripts/aggregate_seeds.py --budget-ratio 0.6 --outdir results/multiseed
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
from typing import Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import dump_json  # noqa: E402


# ---------------------------------------------------------------------------
def fisher_combined_p(pvals: List[float]) -> Optional[float]:
    """Fisher 合并 p 值：X = -2 Σ ln p_i ~ χ²(2k)。"""
    ps = [p for p in pvals if p is not None and 0 < p <= 1]
    if not ps:
        return None
    k = len(ps)
    x = -2.0 * sum(__import__("math").log(p) for p in ps)
    try:
        from scipy.stats import chi2

        return float(chi2.sf(x, 2 * k))
    except Exception:
        # 无 scipy 时的兜底：Wilson–Hilferty 近似
        df = 2 * k
        z = ((x / df) ** (1.0 / 3.0) - (1 - 2.0 / (9 * df))) / (2.0 / (9 * df)) ** 0.5
        return float(0.5 * __import__("math").erfc(z / 2 ** 0.5))


def mean_std(vals: List[float]) -> tuple:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


# ---------------------------------------------------------------------------
def discover_seeds() -> Dict[int, pathlib.Path]:
    """返回 {seed: 结果根目录}。支持 results/seed{N}/ 与旧的扁平 results/。"""
    out: Dict[int, pathlib.Path] = {}
    results = ROOT / "results"
    for d in sorted(results.glob("seed*")):
        if d.is_dir() and d.name[4:].isdigit():
            out[int(d.name[4:])] = d
    flat = results / "sensitivity"
    if flat.is_dir() and list(flat.glob("rank_*.json")):
        out.setdefault(0, results)  # 0 表示"扁平布局（种子未知）"
    return out


def collect(seeds: Dict[int, pathlib.Path], budget_ratio: float) -> List[Dict]:
    rows: List[Dict] = []
    combos = set()
    for root in seeds.values():
        for f in (root / "sensitivity").glob("rank_*.json"):
            stem = f.stem[len("rank_"):]
            backbone, _, shift = stem.partition("_")
            combos.add((backbone, shift))

    for backbone, shift in sorted(combos):
        per_seed = []
        for seed, root in sorted(seeds.items()):
            rk = root / "sensitivity" / f"rank_{backbone}_{shift}.json"
            ar = root / "equal_error" / f"arm_{backbone}_{shift}.json"
            if not rk.exists():
                continue
            d = json.loads(rk.read_text(encoding="utf-8"))
            rc = d.get("rank_correlation", {})
            loss = rc.get("loss", {}) or {}
            rec = {
                "seed": seed,
                "rho_loss": loss.get("rho"),
                "p_loss": loss.get("p_value"),
                "significant": bool(loss.get("significant")),
                "n_layers": d.get("n_layers"),
            }
            if ar.exists():
                arm = json.loads(ar.read_text(encoding="utf-8"))
                arms = arm.get("arms") or []
                if arms:
                    pick = min(arms, key=lambda r: abs(r.get("budget_ratio", 0.6) - budget_ratio))
                    rec["cost"] = pick.get("cost_of_domain_ignorance")
                    rec["rel_cost"] = pick.get("relative_cost_of_ignorance")
                    rec["flipped"] = pick.get("ranking_flipped")
            per_seed.append(rec)

        if not per_seed:
            continue

        rhos = [r["rho_loss"] for r in per_seed]
        costs = [r.get("cost") for r in per_seed]
        rels = [r.get("rel_cost") for r in per_seed]
        ps = [r["p_loss"] for r in per_seed]
        flipped = [r.get("flipped") for r in per_seed if r.get("flipped") is not None]

        rho_m, rho_s = mean_std(rhos)
        cost_m, cost_s = mean_std(costs)
        rel_m, rel_s = mean_std(rels)

        rows.append({
            "backbone": backbone,
            "shift": shift,
            "n_seeds": len(per_seed),
            "n_layers": per_seed[0].get("n_layers"),
            "rho_mean": rho_m,
            "rho_std": rho_s,
            "rho_min": min(rhos) if rhos else None,
            "rho_max": max(rhos) if rhos else None,
            "n_significant": sum(1 for r in per_seed if r["significant"]),
            "fisher_p": fisher_combined_p(ps),
            "cost_mean": cost_m,
            "cost_std": cost_s,
            "rel_cost_mean": rel_m,
            "rel_cost_std": rel_s,
            "n_cost_positive": sum(1 for c in costs if c is not None and c > 0),
            "n_flipped": sum(1 for f in flipped if f),
            "n_flipped_total": len(flipped),
            "per_seed": per_seed,
        })
    return rows


# ---------------------------------------------------------------------------
def f(v, nd=3, width=0):
    if v is None:
        return "-".rjust(width) if width else "-"
    s = f"{v:.{nd}f}"
    return s.rjust(width) if width else s


def render(rows: List[Dict], seeds: Dict[int, pathlib.Path], budget_ratio: float) -> str:
    L = []
    L.append("# 多种子聚合结果")
    L.append("")
    L.append(f"随机种子：{sorted(seeds)}（共 {len(seeds)} 个）  ")
    L.append(f"预算占比：{budget_ratio}  ")
    L.append("")
    L.append("生成命令：`python scripts/aggregate_seeds.py`")
    L.append("")
    L.append("## ρ(loss) 与域无知代价的跨种子统计")
    L.append("")
    hdr = ("| 骨干 | 偏移 | 种子数 | 层数 | **ρ 均值 ± 标准差** | ρ 范围 | "
           "显著种子 | **Fisher 合并 p** | **域无知代价 均值 ± 标准差** | 为正的种子 |")
    L.append(hdr)
    L.append("|" + "---|" * 10)
    for r in rows:
        L.append(
            f"| {r['backbone']} | {r['shift']} | {r['n_seeds']} | {r['n_layers']} | "
            f"**{f(r['rho_mean'])} ± {f(r['rho_std'])}** | "
            f"[{f(r['rho_min'])}, {f(r['rho_max'])}] | "
            f"{r['n_significant']}/{r['n_seeds']} | "
            f"**{f(r['fisher_p'], 4)}** | "
            f"**{f(r['cost_mean'], 4)} ± {f(r['cost_std'], 4)}** | "
            f"{r['n_cost_positive']}/{r['n_seeds']} |")
    L.append("")
    L.append("## 逐种子的 ρ(loss)")
    L.append("")
    seed_ids = sorted(seeds)
    L.append("| 骨干 | 偏移 | " + " | ".join(f"seed {s}" for s in seed_ids) + " |")
    L.append("|" + "---|" * (2 + len(seed_ids)))
    for r in rows:
        by = {p["seed"]: p for p in r["per_seed"]}
        cells = []
        for s in seed_ids:
            p = by.get(s)
            cells.append(f(p["rho_loss"]) if p else "-")
        L.append(f"| {r['backbone']} | {r['shift']} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 怎么读这张表")
    L.append("")
    L.append("1. **Fisher 合并 p** 才是主口径。单个种子 n=10~16 层时统计功效很低，")
    L.append("   合并多个独立种子后如果能达到 p < 0.05，说明效应真实存在，")
    L.append("   只是单次实验不足以检出。")
    L.append("2. **域无知代价的正号一致性**比 ρ 更稳健：看「为正的种子」那一列。")
    L.append("   如果所有种子都为正，结论方向就站得住；")
    L.append("   标准差告诉你幅度有多不确定。")
    L.append("3. **ρ 的范围**列用来判断稳定性：如果 [min, max] 跨过 0，")
    L.append("   说明 ρ 的符号本身都不可靠，此时不应把 ρ 当作结论载体。")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-ratio", type=float, default=0.6)
    ap.add_argument("--outdir", default="results/multiseed")
    args = ap.parse_args()

    seeds = discover_seeds()
    if not seeds:
        print("没找到任何种子结果。先跑 scripts/run_seeds.py")
        return 2
    print(f"发现种子：{sorted(seeds)}")
    for s, p in sorted(seeds.items()):
        n = len(list((p / 'sensitivity').glob('rank_*.json')))
        print(f"  seed {s}: {p.relative_to(ROOT)}  ({n} 个敏感性结果)")

    rows = collect(seeds, args.budget_ratio)
    if not rows:
        print("没有可聚合的结果。")
        return 1

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    md = render(rows, seeds, args.budget_ratio)
    (outdir / "multiseed.md").write_text(md, encoding="utf-8")

    cols = ["backbone", "shift", "n_seeds", "n_layers", "rho_mean", "rho_std",
            "rho_min", "rho_max", "n_significant", "fisher_p",
            "cost_mean", "cost_std", "n_cost_positive", "n_flipped", "n_flipped_total"]
    with (outdir / "multiseed.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    dump_json(outdir / "multiseed.json",
              {"budget_ratio": args.budget_ratio, "seeds": sorted(seeds), "rows": rows})

    # 控制台摘要
    print()
    print("=" * 92)
    print("  跨种子聚合")
    print("=" * 92)
    print(f"  {'backbone':<10} {'shift':<15} {'k':>2} {'rho mean±std':>16} "
          f"{'sig':>5} {'Fisher p':>9} {'cost mean±std':>18} {'+':>4}")
    print("  " + "-" * 88)
    for r in rows:
        print(f"  {r['backbone']:<10} {r['shift']:<15} {r['n_seeds']:>2} "
              f"{f(r['rho_mean'])}±{f(r['rho_std']):<7} "
              f"{r['n_significant']}/{r['n_seeds']:<3} {f(r['fisher_p'],4):>9} "
              f"{f(r['cost_mean'],4)}±{f(r['cost_std'],4):<9} "
              f"{r['n_cost_positive']}/{r['n_seeds']}")
    print()
    print(f"  已写入 {outdir / 'multiseed.md'}")
    print(f"  已写入 {outdir / 'multiseed.csv'}")
    print(f"  已写入 {outdir / 'multiseed.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
