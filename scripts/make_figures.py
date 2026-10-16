"""W15–W16：从已保存的结果生成报告用的四张核心图。

    python scripts/make_figures.py --backbone cldnn --shift D1_snr

读取：
  results/sensitivity/sens_<backbone>_<shift>.npz      ← run_sensitivity.py
  results/equal_error/arm_<backbone>_<shift>.json      ← run_equal_error.py
  results/baseline/<backbone>_metrics.json             ← train_baseline.py（可选）

输出：
  results/figures/fig1_sensitivity_heatmap.png
  results/figures/fig2_rank_scatter.png
  results/figures/fig3_equal_error_arms.png
  results/figures/fig4_proxy_scatter.png
  results/figures/fig5_domain_gap.png            （若 baseline 指标存在）

**本脚本是这四个图表的唯一权威入口。**
不要手动画图 —— 报告里的每张图都必须能由一条命令重生成，这是可复现性的一部分。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.viz.figures import (  # noqa: E402
    plot_domain_gap,
    plot_equal_error_arms,
    plot_proxy_scatter,
    plot_rank_scatter,
    plot_sensitivity_heatmap,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="cldnn")
    ap.add_argument("--shift", default="D1_snr")
    ap.add_argument("--reference-bits", type=int, default=4)
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    tag = f"{args.backbone}_{args.shift}"
    sens_npz = pathlib.Path(f"results/sensitivity/sens_{tag}.npz")
    arm_json = pathlib.Path(f"results/equal_error/arm_{tag}.json")
    base_json = pathlib.Path(f"results/baseline/{args.backbone}_metrics.json")
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not sens_npz.exists():
        print(f"缺少 {sens_npz}，请先跑 scripts/run_sensitivity.py")
        return 2
    z = np.load(sens_npz, allow_pickle=True)
    layer_names = [str(s) for s in z["layer_names"]]
    ladder = [int(b) for b in z["bits_ladder"]]
    ref = int(z["reference_bits"][0]) if "reference_bits" in z else args.reference_bits
    idx = ladder.index(ref)
    dl_a, dl_b = z["delta_loss_a"], z["delta_loss_b"]
    da_a, da_b = z["delta_acc_a"], z["delta_acc_b"]

    made = []
    made.append(plot_sensitivity_heatmap(
        dl_a, dl_b, layer_names, ladder,
        outdir / "fig1_sensitivity_heatmap.png",
        name_a=f"{args.backbone} / Domain A (loss)", 
        name_b=f"{args.backbone} / {args.shift} (loss)"))

    # ρ 与 p 从 rank_*.json 读（图里只做标注，不重算）
    rank_json = pathlib.Path(f"results/sensitivity/rank_{tag}.json")
    rho = p = float("nan")
    if rank_json.exists():
        rk = json.loads(rank_json.read_text(encoding="utf-8"))
        row = rk.get("rank_correlation", {}).get("loss", {})
        rho = row.get("rho", float("nan"))
        p = row.get("p_value", float("nan"))
    made.append(plot_rank_scatter(
        dl_a[:, idx], dl_b[:, idx], layer_names, rho, p,
        outdir / "fig2_rank_scatter.png", reference_bits=ref))

    if arm_json.exists():
        arm = json.loads(arm_json.read_text(encoding="utf-8"))
        arms_all = arm.get("arms", [])
        if arms_all:
            # 取中间的预算档（默认三档 0.4/0.6/0.8）
            pick = sorted(arms_all, key=lambda r: abs(r["budget_ratio"] - 0.6))[0]
            made.append(plot_equal_error_arms(
                {"uniform": pick["uniform_bits"],
                 "agnostic": pick["agnostic_bits"],
                 "oracle": pick["oracle_bits"]},
                {"uniform": pick["uniform_damage"],
                 "agnostic": pick["agnostic_damage"],
                 "oracle": pick["oracle_damage"]},
                ladder, layer_names, outdir / "fig3_equal_error_arms.png",
                budget_ratio=pick["budget_ratio"]))
        proxies = arm.get("proxies", [])
        if proxies:
            # 代理量原始值没有存进 arm json，这里用敏感性做 x 轴的替代展示
            made.append(plot_proxy_scatter(
                {c["proxy"]: np.array([c["rho"]]) for c in proxies},
                np.array([0.0]), proxies, outdir / "fig4_proxy_correlation.png"))
            print("  注意：fig4 画的是代理量的 |ρ| 汇总；"
                  "若要画真正的散点图，需在 run_equal_error.py 里一并保存代理量原始值。")
    else:
        print(f"缺少 {arm_json}，跳过图 3/图 4")

    if base_json.exists():
        b = json.loads(base_json.read_text(encoding="utf-8"))
        doms = b.get("domains", {})
        if doms:
            made.append(plot_domain_gap(
                {k: v["acc"] for k, v in doms.items()},
                outdir / "fig5_domain_gap.png", metric="accuracy"))

    print("\n已生成：")
    for p_ in made:
        print(f"  {p_}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
