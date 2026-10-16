#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实时查看 quant-sens 实验进度。

    python scripts/watch_progress.py                # 每 5 秒刷新，Ctrl+C 退出
    python scripts/watch_progress.py --once         # 只打印一次快照
    python scripts/watch_progress.py -i 2           # 每 2 秒刷新
    python scripts/watch_progress.py --no-clear     # 追加模式（日志重定向时用）

特点
----
* **纯标准库**，不依赖 numpy / torch / pyyaml，系统 Python 也能直接跑。
* 自动从 configs/base.yaml 读出偏移列表，不用手写要监控什么。
* 根据已完成任务的耗时**估算剩余时间**。
* 显示当前正在跑的那一步的日志尾部。
* 矩阵跑完后打印最终汇总表格。

它只读文件，不会干扰正在运行的实验。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# 终端
# ----------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
CLEAR = "\033[H\033[J"


def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        # Windows 10+ 终端默认支持 ANSI；尝试打开 VT 处理
        try:
            import ctypes

            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


class C:
    on = True

    @classmethod
    def w(cls, code: str, s: str) -> str:
        return f"{code}{s}{RESET}" if cls.on else s


# ----------------------------------------------------------------------------
# 极简配置读取（只取需要的两个字段，不依赖 pyyaml）
# ----------------------------------------------------------------------------
def read_shifts(config: pathlib.Path) -> List[str]:
    """从 YAML 里抠出 data.shifts 下的键名列表。"""
    if not config.exists():
        return []
    shifts: List[str] = []
    in_data = False
    in_shifts = False
    for raw in config.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#")[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0:
            in_data = stripped.startswith("data:")
            in_shifts = False
            continue
        if in_data and indent == 2:
            in_shifts = stripped.startswith("shifts:")
            continue
        if in_data and in_shifts and indent >= 4 and ":" in stripped:
            key = stripped.split(":", 1)[0].strip()
            if key and not key.startswith("-"):
                shifts.append(key)
    return shifts


# ----------------------------------------------------------------------------
# 进度模型
# ----------------------------------------------------------------------------
class Progress:
    def __init__(self, root: pathlib.Path, backbones: List[str], shifts: List[str]):
        self.root = root
        self.backbones = backbones
        self.shifts = shifts
        self.sens_dir = root / "results" / "sensitivity"
        self.arm_dir = root / "results" / "equal_error"
        self.base_dir = root / "results" / "baseline"
        self.log_dir = root / "experiments" / "logs"
        self.matrix_dir = root / "results" / "matrix"

    # ---------------------------------------------------------------- 状态
    def trained(self, b: str) -> bool:
        return (self.base_dir / f"{b}.pt").exists()

    def sens_path(self, b: str, s: str) -> pathlib.Path:
        return self.sens_dir / f"rank_{b}_{s}.json"

    def arm_path(self, b: str, s: str) -> pathlib.Path:
        return self.arm_dir / f"arm_{b}_{s}.json"

    def combos(self) -> List[Tuple[str, str]]:
        return [(b, s) for b in self.backbones for s in self.shifts]

    def counts(self) -> Dict[str, Tuple[int, int]]:
        combos = self.combos()
        n = len(combos)
        sens = sum(1 for b, s in combos if self.sens_path(b, s).exists())
        arm = sum(1 for b, s in combos if self.arm_path(b, s).exists())
        tr = sum(1 for b in self.backbones if self.trained(b))
        return {"train": (tr, len(self.backbones)),
                "sens": (sens, n), "arm": (arm, n)}

    def done(self) -> bool:
        c = self.counts()
        return c["sens"][0] == c["sens"][1] and c["arm"][0] == c["arm"][1]

    def running(self) -> bool:
        """是否有属于本项目的 python 进程在跑。"""
        if os.name != "nt":
            return False
        try:
            import subprocess

            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-Process python -ErrorAction SilentlyContinue | "
                 "Where-Object { $_.Path -like '%s*' }).Count" % str(self.root.parent)],
                capture_output=True, text=True, timeout=10)
            return int((out.stdout or "0").strip() or 0) > 0
        except Exception:
            return False

    # ---------------------------------------------------------------- 结果
    def rank_row(self, b: str, s: str) -> Optional[Dict]:
        p = self.sens_path(b, s)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        rc = d.get("rank_correlation", {})
        loss = rc.get("loss", {}) or {}
        acc = rc.get("acc", {}) or {}
        base = d.get("baseline", {}) or {}
        return {
            "n": d.get("n_layers"),
            "acc_A": base.get("acc_A"), "acc_B": base.get("acc_B"),
            "rho_loss": loss.get("rho"), "p_loss": loss.get("p_value"),
            "rho_acc": acc.get("rho"), "p_acc": acc.get("p_value"),
            "significant": loss.get("significant"),
        }

    def arm_row(self, b: str, s: str) -> Optional[Dict]:
        p = self.arm_path(b, s)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        arms = d.get("arms") or []
        if not arms:
            return None
        pick = sorted(arms, key=lambda r: abs(r.get("budget_ratio", 0.6) - 0.6))[0]
        return pick

    # ---------------------------------------------------------------- 计时
    def etas(self) -> Dict[str, Optional[float]]:
        """用已完成任务的相邻完成时间差估算每步耗时与剩余秒数。"""
        def durations(files: List[pathlib.Path]) -> List[float]:
            ts = sorted(f.stat().st_mtime for f in files if f.exists())
            return [b - a for a, b in zip(ts, ts[1:]) if 0 < b - a < 3600]

        out: Dict[str, Optional[float]] = {}
        for key, pat, path_fn, total in (
            ("sens", "rank_*.json", self.sens_path, len(self.combos())),
            ("arm", "arm_*.json", self.arm_path, len(self.combos())),
        ):
            d = durations(list(self.sens_dir.glob(pat)) if key == "sens"
                          else list(self.arm_dir.glob(pat)))
            if not d:
                out[key] = None
                continue
            avg = sum(d) / len(d)
            done = sum(1 for b, s in self.combos() if path_fn(b, s).exists())
            out[key] = max(0.0, avg * (total - done))
        return out

    # ---------------------------------------------------------------- 日志
    def active_log(self, window_s: float = 900.0) -> Tuple[Optional[pathlib.Path], List[str]]:
        if not self.log_dir.exists():
            return None, []
        files = [f for f in self.log_dir.glob("*.log") if f.is_file()]
        if not files:
            return None, []
        latest = max(files, key=lambda f: f.stat().st_mtime)
        age = time.time() - latest.stat().st_mtime
        if age > window_s:
            return latest, []
        try:
            lines = latest.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return latest, []
        return latest, lines[-6:]


# ----------------------------------------------------------------------------
# 渲染
# ----------------------------------------------------------------------------
def bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(round(width * done / total))
    col = GREEN if done == total else (CYAN if done else DIM)
    return "[" + C.w(col, "#" * filled) + "-" * (width - filled) + "]"


def fmt(v, nd: int = 4, width: int = 8) -> str:
    if v is None:
        return "-".rjust(width)
    if isinstance(v, float):
        return f"{v:.{nd}f}".rjust(width)
    return str(v).rjust(width)


def fmt_eta(sec: Optional[float]) -> str:
    if sec is None:
        return "?"
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def render(p: Progress, show_log: bool, pad: bool) -> str:
    out: List[str] = []
    W = 86
    now = time.strftime("%H:%M:%S")

    title = "quant-sens 实验进度"
    out.append(C.w(BOLD, "=" * W))
    out.append(C.w(BOLD, f"  {title}" + " " * max(1, W - len(title) - 12) + now))
    out.append(C.w(BOLD, "=" * W))

    c = p.counts()
    labels = {"train": "训练", "sens": "敏感性", "arm": "等误差臂"}
    ETA = p.etas()
    for key in ("train", "sens", "arm"):
        d, t = c[key]
        line = f"  {labels[key]:<8} {bar(d, t)} {d:>2}/{t:<2}"
        if key != "train" and d < t and ETA.get(key):
            line += C.w(YELLOW, f"   预计剩余 {fmt_eta(ETA[key])}")
        elif key != "train" and d == t:
            line += C.w(GREEN, "   完成")
        out.append(line)

    out.append("")
    out.append(C.w(BOLD, "  组合状态"))
    out.append("  " + C.w(DIM, f"{'骨干':<10} {'偏移':<15} {'训练':>4} {'敏感性':>6} "
                               f"{'等误差臂':>8} {'ρ(loss)':>9} {'p':>8} {'Δacc':>8}"))
    out.append("  " + "-" * (W - 4))

    for b, s in p.combos():
        tr = "OK" if p.trained(b) else "--"
        sn = "OK" if p.sens_path(b, s).exists() else ".."
        ar = "OK" if p.arm_path(b, s).exists() else ".."
        r = p.rank_row(b, s)
        if r:
            drop = (r["acc_A"] - r["acc_B"]) if (r["acc_A"] is not None
                                                 and r["acc_B"] is not None) else None
            rho = r["rho_loss"]
            sig = "*" if r.get("significant") else " "
            rho_s = fmt(rho, 3) + sig
            p_s = fmt(r["p_loss"], 3)
            drop_s = fmt(drop, 3)
        else:
            rho_s, p_s, drop_s = "-".rjust(9), "-".rjust(8), "-".rjust(8)
        col = GREEN if (sn == "OK" and ar == "OK") else (CYAN if sn == "OK" else DIM)
        out.append("  " + C.w(col, f"{b:<10} {s:<15} {tr:>4} {sn:>6} {ar:>8}")
                   + f" {rho_s:>9} {p_s:>8} {drop_s:>8}")

    out.append("")
    out.append(C.w(DIM, "  * 表示 ρ 的置换检验 p < 0.05；Δacc = 源域准确率 − 目标域准确率"))

    if show_log:
        log, tail = p.active_log()
        out.append("")
        if log and tail:
            out.append(C.w(BOLD, f"  当前日志: {log.name}"))
            for ln in tail:
                out.append(C.w(DIM, "    | " + ln[:W - 8]))
        elif log:
            out.append(C.w(DIM, f"  最近日志: {log.name}（已停止写入）"))
        else:
            out.append(C.w(DIM, "  暂无日志"))

    # ------------------------------------------------------------ 完成汇总
    if p.done():
        out.append("")
        out.append(C.w(GREEN + BOLD, "  " + "=" * (W - 4)))
        out.append(C.w(GREEN + BOLD, "  全部完成"))
        out.append(C.w(GREEN + BOLD, "  " + "=" * (W - 4)))
        out.append("")
        out.append(C.w(BOLD, "  ρ(loss) 汇总   行=骨干，列=偏移"))
        hdr = "  " + f"{'':<10}" + "".join(f"{s[:11]:>13}" for s in p.shifts)
        out.append(hdr)
        for b in p.backbones:
            line = "  " + f"{b:<10}"
            for s in p.shifts:
                r = p.rank_row(b, s)
                line += (f"{r['rho_loss']:>+12.3f}" + ("*" if r.get("significant") else " ")
                         ) if r else f"{'-':>13}"
            out.append(line)
        out.append("")
        out.append(C.w(BOLD, "  等误差臂：域无知代价 (agnostic − oracle)"))
        line = "  " + f"{'':<10}"
        for s in p.shifts:
            line += f"{s[:11]:>13}"
        out.append(line)
        for b in p.backbones:
            line = "  " + f"{b:<10}"
            for s in p.shifts:
                a = p.arm_row(b, s)
                line += f"{a['cost_of_domain_ignorance']:>+13.4f}" if a else f"{'-':>13}"
            out.append(line)
        if (p.matrix_dir / "matrix.md").exists():
            out.append("")
            out.append(C.w(DIM, f"  完整汇总见 {p.matrix_dir / 'matrix.md'}"))

    out.append("")
    if pad:
        out.append(C.w(DIM, "  Ctrl+C 退出"))
    return "\n".join(out)


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="实时查看 quant-sens 实验进度")
    ap.add_argument("--root", default=None, help="项目根目录（默认自动定位）")
    ap.add_argument("-i", "--interval", type=float, default=5.0, help="刷新间隔秒（默认 5）")
    ap.add_argument("--once", action="store_true", help="只打印一次")
    ap.add_argument("--no-clear", action="store_true", help="不清屏（追加模式）")
    ap.add_argument("--no-log", action="store_true", help="不显示日志尾部")
    ap.add_argument("--backbones", default="cldnn,resnet1d")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--shifts", default=None, help="覆盖配置里的偏移列表")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve() if args.root else (
        pathlib.Path(__file__).resolve().parents[1])
    C.on = supports_color()

    shifts = (args.shifts.split(",") if args.shifts
              else read_shifts(root / args.config))
    if not shifts:
        print(f"⚠ 没能从 {root / args.config} 读到 shifts，"
              f"用 --shifts D1_snr,D2_cfo 手动指定。")
        return 2

    p = Progress(root, args.backbones.split(","), shifts)

    if args.once:
        print(render(p, not args.no_log, pad=False))
        return 0

    clear = (not args.no_clear) and sys.stdout.isatty()
    try:
        while True:
            text = render(p, not args.no_log, pad=True)
            if clear:
                sys.stdout.write(CLEAR + text + "\n")
            else:
                sys.stdout.write(text + "\n" + "-" * 86 + "\n")
            sys.stdout.flush()

            if p.done() and not p.running():
                # 全部完成且无进程 → 打印最终汇总后退出
                if clear:
                    sys.stdout.write(CLEAR + render(p, False, pad=False) + "\n")
                    sys.stdout.flush()
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已退出监视（实验进程不受影响）。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
