"""ρ 对**参考位宽**的依赖性分析。

为什么需要它
------------
逐层敏感性是在某一个"参考位宽" b* 上测量的。b* 选得不好会毁掉秩分析：

* b* 太大（如 8 bit）→ 几乎每一层的 Δloss 都 ≈ 0 → **大量并列值** →
  秩相关由近似随机的微小差异决定；
* b* 太小（如 2 bit）→ 所有层都严重退化 → 差异被压缩、同样难分；
* 中间某一档才可能给出有区分度的排序。

实测（CLDNN, D4_phasenoise）：参考位宽 4 bit 时 ρ=+0.139 (p=0.71)，
但域间隙其实很大（acc 0.965→0.540）。这说明"ρ 低"可能只是位宽选得不好，
而不是"排序真的不可迁移"。**这两种解释的结论完全相反**，必须区分开。

用法
----
    python scripts/rank_vs_bits.py                      # 扫描 results/sensitivity 下全部结果
    python scripts/rank_vs_bits.py --shift D4_phasenoise
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.rank_corr import analyse_pair  # noqa: E402
from src.utils import dump_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=None, help="只看某个骨干；默认全部")
    ap.add_argument("--shift", default=None, help="只看某个偏移；默认全部")
    ap.add_argument("--n-permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results/rank_vs_bits")
    args = ap.parse_args()

    sens_dir = ROOT / "results" / "sensitivity"
    files = sorted(sens_dir.glob("sens_*.npz"))
    if not files:
        print(f"没找到 {sens_dir}/sens_*.npz，请先跑 scripts/run_sensitivity.py")
        return 2

    rows = []
    for f in files:
        stem = f.stem[len("sens_"):]
        backbone, _, shift = stem.partition("_")
        if args.backbone and backbone != args.backbone:
            continue
        if args.shift and shift != args.shift:
            continue

        z = np.load(f, allow_pickle=True)
        ladder = [int(b) for b in z["bits_ladder"]]
        layer_names = [str(s) for s in z["layer_names"]]
        dl_a, dl_b = z["delta_loss_a"], z["delta_loss_b"]
        da_a, da_b = z["delta_acc_a"], z["delta_acc_b"]

        print(f"\n{'=' * 78}")
        print(f"  {backbone} × {shift}   层数 n={len(layer_names)}   位宽阶梯 {ladder}")
        print(f"{'=' * 78}")
        print(f"  {'ref_bits':>8} | {'rho_loss':>9} {'p_loss':>8} "
              f"{'#uniq_A':>8} {'#uniq_B':>8} | {'rho_acc':>8} {'p_acc':>8}")
        print("  " + "-" * 72)

        for j, b in enumerate(ladder):
            sa, sb = dl_a[:, j], dl_b[:, j]
            ta, tb = da_a[:, j], da_b[:, j]

            na, nb = int(np.unique(sa).size), int(np.unique(sb).size)
            ta_u, tb_u = int(np.unique(ta).size), int(np.unique(tb).size)

            rl = analyse_pair(sa, sb, n_permutations=args.n_permutations,
                              seed=args.seed, n_boot=0)
            ra = analyse_pair(ta, tb, n_permutations=args.n_permutations,
                              seed=args.seed, n_boot=0)

            # 并列太多时 ρ 不可信，标出来
            tie_flag = "  <-- 并列过多" if (na < len(layer_names) * 0.8
                                            or nb < len(layer_names) * 0.8) else ""
            print(f"  {b:>8} | {rl.rho:>+9.4f} {rl.p_value:>8.4f} "
                  f"{na:>8} {nb:>8} | {ra.rho:>+8.4f} {ra.p_value:>8.4f}{tie_flag}")

            rows.append({
                "backbone": backbone, "shift": shift, "ref_bits": b,
                "n_layers": len(layer_names),
                "rho_loss": rl.rho, "p_loss": rl.p_value,
                "ci_loss": [rl.ci_low, rl.ci_high],
                "n_unique_loss_A": na, "n_unique_loss_B": nb,
                "rho_acc": ra.rho, "p_acc": ra.p_value,
                "n_unique_acc_A": ta_u, "n_unique_acc_B": tb_u,
                "significant_loss": rl.significant,
                "verdict_loss": rl.verdict(),
            })

        # 给出该组合下"最可信"的参考位宽：并列最少且灵敏度最高
        sub = [r for r in rows if r["backbone"] == backbone and r["shift"] == shift]
        if sub:
            best = max(sub, key=lambda r: (min(r["n_unique_loss_A"], r["n_unique_loss_B"]),
                                           abs(r["rho_loss"])))
            print(f"\n  并列最少的参考位宽 = {best['ref_bits']} bit  "
                  f"(唯一值 A={best['n_unique_loss_A']}, B={best['n_unique_loss_B']}, "
                  f"ρ={best['rho_loss']:+.4f}, p={best['p_loss']:.4f})")

    if not rows:
        print("没有匹配的结果。")
        return 1

    # ---------------------------------------------------------------- 汇总
    print(f"\n{'=' * 78}")
    print("  汇总：每个 (骨干, 偏移) 在各参考位宽下的 ρ(loss)")
    print(f"{'=' * 78}")
    keys = sorted({(r["backbone"], r["shift"]) for r in rows})
    ladder_all = sorted({r["ref_bits"] for r in rows})
    header = f"  {'backbone':<10} {'shift':<15} " + "".join(
        f"{str(b) + 'b':>10}" for b in ladder_all)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for bb, sh in keys:
        line = f"  {bb:<10} {sh:<15} "
        for b in ladder_all:
            m = [r for r in rows if r["backbone"] == bb and r["shift"] == sh
                 and r["ref_bits"] == b]
            if m:
                star = "*" if m[0]["significant_loss"] else " "
                line += f"{m[0]['rho_loss']:>+9.3f}{star}"
            else:
                line += f"{'-':>10}"
        print(line)
    print("\n  * 表示 p < 0.05。空位表示该位宽不在该组合的阶梯里。")

    out = dump_json(ROOT / args.outdir / "rank_vs_bits.json", {
        "n_permutations": args.n_permutations, "seed": args.seed, "rows": rows,
    })
    print(f"\n  已写入 {out}")
    print("\n  ⚠ 读法：若不同参考位宽给出的 ρ 差异很大（例如 4b 是 +0.14、2b 是 +0.7），")
    print("    说明**参考位宽本身是一个需要论证的设计选择**，")
    print("    必须在报告里说明为什么选它，并给出对位宽的敏感性分析。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
