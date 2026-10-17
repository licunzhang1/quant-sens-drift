#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一次性文档补丁：把多种子结果反映到 README 与 docs/findings.md。

之所以用 Python 而不是 PowerShell：中文经 PowerShell -Command 传递时会被
破坏成乱码，导致解析错误。Python 脚本文件本身是 UTF-8，可靠。
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def patch(path: pathlib.Path, pairs: list[tuple[str, str]], must_all: bool = False) -> int:
    old_text = path.read_text(encoding="utf-8")
    text = old_text
    hits = 0
    for old, new in pairs:
        if old and old in text:
            text = text.replace(old, new, 1)
            hits += 1
        elif must_all:
            print(f"  MISS in {path.name}: {old[:60]!r}")
    if text != old_text:
        path.write_text(text, encoding="utf-8")
    return hits


def main() -> int:
    findings = ROOT / "docs" / "findings.md"
    readme = ROOT / "README.md"

    # ---------------------------------------------------------------- findings
    t = findings.read_text(encoding="utf-8")
    changed = 0

    warn9 = (
        "> **注意**：本节全部 ρ 均为单种子（seed=42）的点估计。三种子复核见发现 10 ——\n"
        "> 其中 D4 相位噪声与 D5 IQ 失衡的 ρ **符号发生翻转**。\n"
    )
    h9 = "## 发现 9｜完整实验矩阵：2 骨干 × 5 偏移（含阴性对照）"
    if h9 in t and "本节全部 ρ 均为单种子" not in t:
        t = t.replace(h9, h9 + "\n\n" + warn9, 1)
        changed += 1

    warn8 = (
        "> **注意**：本节的 ρ 同样是单种子结果。多种子复核（发现 10）显示 ρ 的\n"
        "> **符号本身不稳定**，因此下表数值只能读作“该次抽样的一个样本”。\n"
    )
    h8 = "## 发现 8｜参考位宽是**一阶设计选择**，不是细节"
    if h8 in t and "本节的 ρ 同样是单种子结果" not in t:
        t = t.replace(h8, h8 + "\n\n" + warn8, 1)
        changed += 1

    row = "| **图表生成** | `scripts/make_figures.py` | 通过，PNG 已渲染 |"
    new_rows = (
        row
        + "\n| **多种子实验（3 种子 × 2 骨干 × 5 偏移）** | `scripts/run_seeds.py` "
          "| **通过**，44/44 单元，135 分钟 |"
        + "\n| **跨种子聚合（均值±标准差 + Fisher 合并）** | `scripts/aggregate_seeds.py` "
          "| **通过** |"
    )
    if row in t and "run_seeds.py` | **通过**" not in t:
        t = t.replace(row, new_rows, 1)
        changed += 1

    if t != findings.read_text(encoding="utf-8"):
        findings.write_text(t, encoding="utf-8")
    print(f"findings.md: {changed} 处")

    # ---------------------------------------------------------------- README
    t = readme.read_text(encoding="utf-8")
    changed = 0

    anchor = "汇总产物：`results/matrix/matrix.md`、`matrix.csv`、`summary.json`。"
    addition = anchor + """

### 多种子（给出误差棒）

单次运行的 ρ 方差很大，只有多种子才能给出“均值 ± 标准差”。

```bash
# 补跑额外种子（可断点续跑）；两个种子约 135 分钟（CPU）
python scripts/run_seeds.py --seeds 43,44 --skip-existing

# 跑的时候实时看进度（另开一个终端，纯标准库）
python scripts/watch_seeds.py --seeds 43,44

# 聚合：均值±标准差 + 跨种子 Fisher 合并 p
python scripts/aggregate_seeds.py
```

产物：`results/multiseed/multiseed.md`、`multiseed.csv`、`multiseed.json`。
结果与订正说明见 [docs/findings.md](docs/findings.md) 发现 10。"""
    if anchor in t and "多种子（给出误差棒）" not in t:
        t = t.replace(anchor, addition, 1)
        changed += 1

    d_old = "scripts/       训练、敏感性、等误差臂、矩阵、出图、进度查看"
    d_new = "scripts/       训练、敏感性、等误差臂、矩阵、多种子、聚合、出图、进度查看"
    if d_old in t:
        t = t.replace(d_old, d_new, 1)
        changed += 1

    if t != readme.read_text(encoding="utf-8"):
        readme.write_text(t, encoding="utf-8")
    print(f"README.md:  {changed} 处")

    # ---------------------------------------------------------------- 自检
    print()
    print("--- findings.md 章节 ---")
    for line in findings.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            print("  " + line)
    print()
    print(f"findings.md {findings.stat().st_size/1024:.1f} KB   "
          f"README.md {readme.stat().st_size/1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
