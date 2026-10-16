"""一键跑完全部验证。

    python tests/run_all.py            # 单元测试 + 冒烟测试
    python tests/run_all.py --quick    # 只跑单元测试（几秒）

退出码 0 表示全部通过。**每次改完核心代码都应该跑一遍这个。**
"""

from __future__ import annotations

# --- console safety --------------------------------------------------------
# Chinese Windows consoles default to GBK. A character outside GBK raises
# UnicodeEncodeError on print and aborts the whole script. Setting
# errors="replace" degrades such a character to "?" instead of crashing.
import sys as _console_sys

for _stream in (_console_sys.stdout, _console_sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass
del _console_sys, _stream
# --- end console safety ----------------------------------------------------

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def run(script: pathlib.Path, extra: list[str] | None = None) -> tuple[int, float]:
    cmd = [sys.executable, str(script)] + (extra or [])
    print(f"\n{'#' * 78}\n#  {script.name}\n{'#' * 78}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(HERE.parent))
    return proc.returncode, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑单元测试")
    args = ap.parse_args()

    plan = [("test_quant_core.py", None)]
    if not args.quick:
        plan.append(("smoke_pipeline.py", None))

    results = []
    for name, extra in plan:
        rc, dt = run(HERE / name, extra)
        results.append((name, rc, dt))

    print(f"\n{'=' * 78}")
    print("  汇总")
    print(f"{'=' * 78}")
    bad = 0
    for name, rc, dt in results:
        mark = "PASS" if rc == 0 else "FAIL"
        if rc != 0:
            bad += 1
        print(f"  [{mark}] {name:<24s} {dt:>7.1f}s  (exit={rc})")
    print()
    if bad:
        print(f"  {bad} 个脚本失败")
    else:
        print("  全部通过")
        print()
        print("  本脚本覆盖的是**不依赖 torch 的科学核心**（量化数学、秩相关、")
        print("  等误差臂、端到端管线）。torch 侧另有一套验证：")
        print("    python scripts/train_baseline.py --config configs/base.yaml")
        print("    python scripts/run_matrix.py --skip-existing")
        print("  实测结果见 docs/findings.md。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
