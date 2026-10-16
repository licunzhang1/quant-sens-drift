"""跑完整实验矩阵：骨干 × 分布偏移，汇总成一张表。

    # 先看看要跑什么（不执行）
    python scripts/run_matrix.py --dry-run

    # 跑全部组合（骨干用 configs 里能认的，偏移取 config 的 data.shifts 子集）
    python scripts/run_matrix.py --backbones cldnn,resnet1d --shifts D1_snr,D2_cfo,D3_gain

    # 顺带把缺的骨干权重也训出来
    python scripts/run_matrix.py --train-missing

产物
----
results/matrix/matrix.csv        汇总表（可直接贴进报告）
results/matrix/matrix.md         Markdown 版
results/matrix/summary.json      完整结果

为什么要有这个脚本
------------------
单次运行的 ρ 方差很大（n=10~16 层时 95% CI 仍然很宽）。本项目的结论
**必须靠多骨干 × 多偏移累积证据**，而不是看某一次的 ρ 是否显著。
这个脚本把"累积证据"变成一条命令。

（见 docs/findings.md：冒烟测试得到 ρ=+0.71、p=0.06，
  正落在"接近显著但未达"的区间 —— 单次结果不足以支撑结论。）
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import dump_json, load_config  # noqa: E402


def run(cmd: List[str], log: pathlib.Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"    $ {' '.join(cmd[1:])}")
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"    ✗ 退出码 {proc.returncode}，日志 {log}")
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-12:]
        for line in tail:
            print(f"      | {line}")
    return proc.returncode


def collect_row(backbone: str, shift: str) -> Optional[Dict]:
    rank_json = ROOT / "results" / "sensitivity" / f"rank_{backbone}_{shift}.json"
    arm_json = ROOT / "results" / "equal_error" / f"arm_{backbone}_{shift}.json"
    if not rank_json.exists():
        return None
    rk = json.loads(rank_json.read_text(encoding="utf-8"))
    row: Dict[str, object] = {
        "backbone": backbone,
        "shift": shift,
        "n_layers": rk.get("n_layers"),
        "ref_bits": rk.get("reference_bits"),
        "acc_A": (rk.get("baseline") or {}).get("acc_A"),
        "acc_B": (rk.get("baseline") or {}).get("acc_B"),
        "loss_A": (rk.get("baseline") or {}).get("loss_A"),
        "loss_B": (rk.get("baseline") or {}).get("loss_B"),
    }
    rc = rk.get("rank_correlation", {})
    for key in ("loss", "acc"):
        d = rc.get(key) or {}
        row[f"rho_{key}"] = d.get("rho")
        row[f"p_{key}"] = d.get("p_value")
        row[f"ci_{key}"] = f"[{d.get('ci_low')}, {d.get('ci_high')}]"
    row["verdict_loss"] = (rc.get("loss") or {}).get("verdict", "")

    if arm_json.exists():
        arm = json.loads(arm_json.read_text(encoding="utf-8"))
        arms = arm.get("arms") or []
        if arms:
            pick = sorted(arms, key=lambda r: abs(r["budget_ratio"] - 0.6))[0]
            row["budget_ratio"] = pick["budget_ratio"]
            row["cost_of_ignorance"] = pick["cost_of_domain_ignorance"]
            row["rel_cost_of_ignorance"] = pick["relative_cost_of_ignorance"]
            row["gain_from_sensitivity"] = pick["gain_from_sensitivity"]
            row["ranking_flipped"] = pick["ranking_flipped"]
            row["verdict_arm"] = pick["verdict"]
    return row


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--backbones", default=None, help="逗号分隔；默认 cldnn,resnet1d")
    ap.add_argument("--shifts", default=None, help="逗号分隔；默认 config 里全部 shifts")
    ap.add_argument("--train-missing", action="store_true", help="缺权重时先训练")
    ap.add_argument("--skip-existing", action="store_true",
                    help="跳过已有结果的组合（断点续跑用；中断后可放心重跑本命令）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outdir", default="results/matrix")
    args = ap.parse_args()

    cfg = load_config(args.config)
    backbones = (args.backbones.split(",") if args.backbones
                 else ["cldnn", "resnet1d"])
    all_shifts = list((cfg["data"].get("shifts") or {}).keys())
    shifts = (args.shifts.split(",") if args.shifts else all_shifts)

    py = sys.executable
    plan = [(b, s) for b in backbones for s in shifts]

    def sens_done(b: str, s: str) -> bool:
        return (ROOT / "results" / "sensitivity" / f"rank_{b}_{s}.json").exists()

    def arm_done(b: str, s: str) -> bool:
        return (ROOT / "results" / "equal_error" / f"arm_{b}_{s}.json").exists()

    print(f"[run_matrix] 配置={args.config}")
    print(f"  骨干  : {backbones}")
    print(f"  偏移  : {shifts}")
    print(f"  组合数: {len(plan)}")
    if args.skip_existing:
        done = sum(1 for b, s in plan if sens_done(b, s) and arm_done(b, s))
        print(f"  --skip-existing: 已完整 {done} 个，待跑 {len(plan) - done} 个")
    if args.dry_run:
        for b, s in plan:
            ck = ROOT / "results" / "baseline" / f"{b}.pt"
            print(f"    {b:<10s} × {s:<14s}  "
                  f"权重{'已存在' if ck.exists() else '缺失'}  "
                  f"敏感性{'已完成' if sens_done(b, s) else '待跑'}  "
                  f"等误差臂{'已完成' if arm_done(b, s) else '待跑'}")
        return 0

    logs = ROOT / "experiments" / "logs"
    rc_all = 0

    # ---------------------------------------------------------------- 训练
    for b in backbones:
        ck = ROOT / "results" / "baseline" / f"{b}.pt"
        if ck.exists():
            print(f"\n[train] {b}: 权重已存在，跳过")
            continue
        if not args.train_missing:
            print(f"\n[train] {b}: 缺权重 {ck}，用 --train-missing 才会训练。跳过。")
            rc_all = 1
            continue
        print(f"\n[train] {b}")
        rc_all |= run([py, "scripts/train_baseline.py", "--config", args.config,
                       "--backbone", b], logs / f"train_{b}.log")

    # ---------------------------------------------------------------- 敏感性
    print("\n[sensitivity]")
    for b, s in plan:
        if args.skip_existing and sens_done(b, s):
            print(f"    {b} × {s}: 已完成，跳过")
            continue
        rc = run([py, "scripts/run_sensitivity.py", "--config", args.config,
                  "--shift", s, "--backbone", b], logs / f"sens_{b}_{s}.log")
        rc_all |= rc

    # ---------------------------------------------------------------- 等误差臂
    print("\n[equal_error]")
    for b, s in plan:
        if args.skip_existing and arm_done(b, s):
            print(f"    {b} × {s}: 已完成，跳过")
            continue
        rc = run([py, "scripts/run_equal_error.py", "--config", args.config,
                  "--shift", s, "--backbone", b], logs / f"arm_{b}_{s}.log")
        rc_all |= rc

    # ---------------------------------------------------------------- 汇总
    rows = [r for r in (collect_row(b, s) for b, s in plan) if r]
    if not rows:
        print("\n没有收集到任何结果，检查上面的日志。")
        return 1

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    cols = ["backbone", "shift", "n_layers", "acc_A", "acc_B",
            "rho_loss", "p_loss", "ci_loss", "rho_acc", "p_acc",
            "cost_of_ignorance", "rel_cost_of_ignorance", "ranking_flipped"]

    import csv

    with (outdir / "matrix.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = ["# 实验矩阵汇总", "",
             "> 由 `scripts/run_matrix.py` 自动生成。ρ 为逐层量化敏感性的 Spearman 秩相关；",
             "> `cost_of_ignorance` = 按源域敏感性分配位宽相对按目标域分配的额外预测损伤。", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")

    lines += ["", "## 结论", ""]
    for r in rows:
        lines.append(f"- **{r['backbone']} × {r['shift']}**："
                     f"ρ(loss)={fmt(r.get('rho_loss'))} "
                     f"(p={fmt(r.get('p_loss'))})；{r.get('verdict_loss', '')}")
        if r.get("verdict_arm"):
            lines.append(f"  - 等误差臂：{r['verdict_arm']}")

    (outdir / "matrix.md").write_text("\n".join(lines), encoding="utf-8")
    dump_json(outdir / "summary.json", {"config": args.config, "rows": rows})

    print("\n" + "=" * 78)
    print("  实验矩阵汇总")
    print("=" * 78)
    print("  " + " | ".join(f"{c}" for c in
                           ["backbone", "shift", "n", "rho_loss", "p_loss",
                            "cost_ign", "rel_cost"]))
    for r in rows:
        print("  " + " | ".join([
            str(r["backbone"]), str(r["shift"]), str(r.get("n_layers", "-")),
            fmt(r.get("rho_loss")), fmt(r.get("p_loss")),
            fmt(r.get("cost_of_ignorance"), 6),
            fmt(r.get("rel_cost_of_ignorance"), 3),
        ]))
    print()
    print(f"  已写入 {outdir / 'matrix.csv'}")
    print(f"  已写入 {outdir / 'matrix.md'}")
    print(f"  已写入 {outdir / 'summary.json'}")
    if rc_all:
        print("\n  ⚠ 有子步骤失败，详见 experiments/logs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
