#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多种子实验的实时进度查看器。

    python scripts/watch_seeds.py               # 每 5 秒刷新，Ctrl+C 退出
    python scripts/watch_seeds.py --once        # 只打一次快照
    python scripts/watch_seeds.py -i 2          # 每 2 秒刷新
    python scripts/watch_seeds.py --no-clear    # 追加模式（重定向到文件时用）

只看文件，不干扰正在跑的实验。纯标准库，不需要 numpy / torch。
全部完成后会自动打印跨种子汇总表。

目录约定（与 scripts/run_seeds.py 一致）：
    results/seed{N}/baseline/{backbone}.pt
    results/seed{N}/sensitivity/rank_{backbone}_{shift}.json
    results/seed{N}/equal_error/arm_{backbone}_{shift}.json
"""

import argparse
import json
import os
import pathlib
import sys
import time

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

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, CYAN, RED = "\033[32m", "\033[33m", "\033[36m", "\033[31m"
CLEAR = "\033[H\033[J"
W = 92


class C:
    on = True

    @classmethod
    def w(cls, code, s):
        return f"{code}{s}{RESET}" if cls.on else s


def use_color():
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        try:
            import ctypes

            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


# ---------------------------------------------------------------------------
def read_config(path):
    """只抠出 data.shifts 的键名，不依赖 pyyaml。"""
    shifts, in_data, in_shifts = [], False, False
    if not path.exists():
        return shifts
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#")[0].rstrip()
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip())
        st = line.strip()
        if ind == 0:
            in_data, in_shifts = st.startswith("data:"), False
            continue
        if in_data and ind == 2:
            in_shifts = st.startswith("shifts:")
            continue
        if in_data and in_shifts and ind >= 4 and ":" in st:
            shifts.append(st.split(":", 1)[0].strip())
    return shifts


# ---------------------------------------------------------------------------
class Model:
    def __init__(self, root, backbones, shifts, seed_filter=None):
        self.root = root
        self.backbones = backbones
        self.shifts = shifts
        self.seed_filter = set(seed_filter) if seed_filter else None
        self.results = root / "results"
        self.logs = root / "experiments" / "logs"

    def seeds(self):
        """返回 [(seed, 结果根目录)]。

        ``--seeds`` 指定的种子即使目录还没被创建也会列出来（run_seeds.py 是
        惰性创建目录的），否则进度条的分母会偏小、ETA 会低估一半。
        """
        found = {}
        for d in sorted(self.results.glob("seed*")):
            if d.is_dir() and d.name[4:].isdigit():
                found[int(d.name[4:])] = d
        if self.seed_filter:
            found = {n: d for n, d in found.items() if n in self.seed_filter}
            for n in self.seed_filter:
                found.setdefault(n, self.results / f"seed{n}")
        return sorted(found.items())

    def paths(self, sroot, b, s):
        return (sroot / "baseline" / f"{b}.pt",
                sroot / "sensitivity" / f"rank_{b}_{s}.json",
                sroot / "equal_error" / f"arm_{b}_{s}.json")

    def units(self):
        """返回 [(seed, backbone, shift, kind)] 的全部工作单元。"""
        u = []
        for seed, sroot in self.seeds():
            for b in self.backbones:
                u.append((seed, b, None, "train"))
            for b in self.backbones:
                for s in self.shifts:
                    u.append((seed, b, s, "sens"))
                    u.append((seed, b, s, "arm"))
        return u

    def done(self, seed, sroot, b, s, kind):
        ck, rk, ar = self.paths(sroot, b, s)
        return {"train": ck, "sens": rk, "arm": ar}[kind].exists()

    def counts(self):
        tot = {"train": 0, "sens": 0, "arm": 0}
        don = {"train": 0, "sens": 0, "arm": 0}
        for seed, sroot in self.seeds():
            for b in self.backbones:
                tot["train"] += 1
                don["train"] += self.done(seed, sroot, b, None, "train")
                for s in self.shifts:
                    for k in ("sens", "arm"):
                        tot[k] += 1
                        don[k] += self.done(seed, sroot, b, s, k)
        return don, tot

    def rank(self, sroot, b, s):
        p = sroot / "sensitivity" / f"rank_{b}_{s}.json"
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        rc = d.get("rank_correlation", {}).get("loss", {}) or {}
        base = d.get("baseline", {}) or {}
        return {"rho": rc.get("rho"), "p": rc.get("p_value"),
                "sig": rc.get("significant"), "n": d.get("n_layers"),
                "accA": base.get("acc_A"), "accB": base.get("acc_B")}

    def active_log(self, window=1800):
        if not self.logs.exists():
            return None, []
        fs = [f for f in self.logs.glob("*.log") if f.is_file()]
        if not fs:
            return None, []
        last = max(fs, key=lambda f: f.stat().st_mtime)
        if time.time() - last.stat().st_mtime > window:
            return last, []
        try:
            return last, last.read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
        except Exception:
            return last, []

    def eta(self):
        """剩余时间估计。

        训练单元（约 8 分钟）与敏感性/等误差臂单元（约 1 分钟）耗时差一个数量级，
        用统一平均值会严重失真，所以分开估。常数取自实测：
        cldnn 训练 8.4 分钟、resnet1d 约 7.7 分钟；10 组「敏感性 + 等误差臂」
        合计 22 分钟 → 每组约 2.2 分钟。

        注意：若不做种子过滤，已存在的旧种子会被算进"已完成"，
        ETA 会显著低估。用 --seeds 限定本次运行的种子。
        """
        don, tot = self.counts()
        T_TRAIN = 8.0     # 分钟 / 次训练
        T_COMBO = 2.2     # 分钟 / (一次敏感性 + 一次等误差臂)
        left_train = max(0, tot["train"] - don["train"])
        left_pairs = max(0, tot["sens"] - don["sens"])
        return left_train * T_TRAIN + left_pairs * T_COMBO, None

    def running(self):
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


# ---------------------------------------------------------------------------
def bar(d, t, width=22):
    if t <= 0:
        return "[" + "-" * width + "]"
    k = int(round(width * d / t))
    col = GREEN if d == t else (CYAN if d else DIM)
    return "[" + C.w(col, "#" * k) + "-" * (width - k) + "]"


def f(v, nd=3, w=0):
    if v is None:
        return "-".rjust(w) if w else "-"
    if isinstance(v, float):
        s = f"{v:.{nd}f}"
    else:
        s = str(v)
    return s.rjust(w) if w else s


def hhmmss(sec):
    if sec is None:
        return "?"
    sec = int(sec)
    return f"{sec}s" if sec < 60 else (f"{sec//60}m{sec%60:02d}s" if sec < 3600
                                       else f"{sec//3600}h{(sec%3600)//60:02d}m")


def render(m, show_log, el_avg):
    L = []
    L.append(C.w(BOLD, "=" * W))
    title = "quant-sens  multi-seed progress"
    L.append(C.w(BOLD, f"  {title}" + " " * max(1, W - len(title) - 12) + time.strftime("%H:%M:%S")))
    L.append(C.w(BOLD, "=" * W))

    don, tot = m.counts()
    for k, label in (("train", "train"), ("sens", "sensitivity"), ("arm", "equal-err")):
        line = f"  {label:<12} {bar(don[k], tot[k])} {don[k]:>3}/{tot[k]:<3}"
        if tot[k] and don[k] == tot[k]:
            line += C.w(GREEN, "  done")
        L.append(line)

    eta_min, avg = m.eta()
    if eta_min is not None and sum(tot.values()) > sum(don.values()):
        L.append("")
        L.append(C.w(YELLOW, f"  ETA ~{eta_min:.0f} min  ({sum(tot.values())-sum(don.values())} "
                             f"units left; train 8min each, sens+arm 2.2min each)"))

    L.append("")
    L.append(C.w(BOLD, "  per seed"))
    L.append("  " + C.w(DIM, f"{'seed':>5} {'train':>7} {'sens':>7} {'arm':>7} {'rho(loss) per shift':<40}"))
    L.append("  " + "-" * (W - 4))
    for seed, sroot in m.seeds():
        tr = sum(m.done(seed, sroot, b, None, "train") for b in m.backbones)
        sn = sum(m.done(seed, sroot, b, s, "sens") for b in m.backbones for s in m.shifts)
        ar = sum(m.done(seed, sroot, b, s, "arm") for b in m.backbones for s in m.shifts)
        ttot = len(m.backbones)
        stot = len(m.backbones) * len(m.shifts)
        cells = []
        for s in m.shifts:
            vals = [m.rank(sroot, b, s) for b in m.backbones]
            vals = [v for v in vals if v]
            if vals:
                cells.append(f"{s[:6]}:{f(sum(v['rho'] for v in vals)/len(vals),2)}")
            else:
                cells.append(f"{s[:6]}:.")
        col = GREEN if (tr == ttot and sn == stot and ar == stot) else CYAN
        L.append("  " + C.w(col, f"{seed:>5} {tr:>3}/{ttot:<3} {sn:>3}/{stot:<3} {ar:>3}/{stot:<3} ")
                 + " ".join(cells))

    L.append("")
    L.append(C.w(BOLD, "  detail  (rho = Spearman of per-layer sensitivity ranks)"))
    L.append("  " + C.w(DIM, f"{'seed':>5} {'backbone':<10} {'shift':<15} {'n':>3} "
                               f"{'rho':>8} {'p':>8} {'accA':>7} {'accB':>7}"))
    L.append("  " + "-" * (W - 4))
    for seed, sroot in m.seeds():
        for b in m.backbones:
            for s in m.shifts:
                r = m.rank(sroot, b, s)
                if not r:
                    continue
                star = "*" if r.get("sig") else " "
                L.append("  " + f"{seed:>5} {b:<10} {s:<15} {f(r['n'],0,3)} "
                               f"{f(r['rho'],3,8)}{star} {f(r['p'],3,8)} "
                               f"{f(r['accA'],3,7)} {f(r['accB'],3,7)}")

    if show_log:
        log, tail = m.active_log()
        L.append("")
        if log and tail:
            L.append(C.w(BOLD, f"  current log: {log.name}"))
            for ln in tail:
                L.append(C.w(DIM, "    | " + ln[:W - 8]))
        elif log:
            L.append(C.w(DIM, f"  last log: {log.name} (idle)"))
        else:
            L.append(C.w(DIM, "  no logs yet"))

    done_all = all(don[k] == tot[k] for k in tot) and sum(tot.values()) > 0
    if done_all:
        L.append("")
        L.append(C.w(GREEN + BOLD, "  " + "=" * (W - 4)))
        L.append(C.w(GREEN + BOLD, "  ALL DONE"))
        L.append(C.w(GREEN + BOLD, "  " + "=" * (W - 4)))
        agg = m.results / "multiseed" / "multiseed.json"
        if agg.exists():
            try:
                d = json.loads(agg.read_text(encoding="utf-8"))
                L.append("")
                L.append(C.w(BOLD, f"  aggregate over seeds {d.get('seeds')}"))
                L.append("  " + C.w(DIM, f"{'backbone':<10} {'shift':<15} {'k':>2} "
                                           f"{'rho mean+-std':>18} {'sig':>5} {'Fisher p':>9} "
                                           f"{'cost mean+-std':>20} {'+':>5}"))
                L.append("  " + "-" * (W - 4))
                for r in d.get("rows", []):
                    L.append("  " + f"{r['backbone']:<10} {r['shift']:<15} {r['n_seeds']:>2} "
                                   f"{f(r['rho_mean'])}±{f(r['rho_std']):<8} "
                                   f"{r['n_significant']}/{r['n_seeds']:<3} "
                                   f"{f(r['fisher_p'],4):>9} "
                                   f"{f(r['cost_mean'],4)}±{f(r['cost_std'],4):<10} "
                                   f"{r['n_cost_positive']}/{r['n_seeds']}")
            except Exception as exc:
                L.append(C.w(DIM, f"  (failed to read aggregate: {exc})"))
        else:
            L.append("")
            L.append(C.w(YELLOW, "  next: python scripts/aggregate_seeds.py"))
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="multi-seed progress viewer")
    ap.add_argument("--root", default=None)
    ap.add_argument("-i", "--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-clear", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--backbones", default="cldnn,resnet1d")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--shifts", default=None)
    ap.add_argument("--seeds", default=None,
                    help="只显示这些种子（逗号分隔）。不指定则显示全部；"
                         "跑多种子时建议指定，否则旧种子的已完成工作会让 ETA 低估。")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve() if args.root else pathlib.Path(__file__).resolve().parents[1]
    C.on = use_color()
    shifts = args.shifts.split(",") if args.shifts else read_config(root / args.config)
    if not shifts:
        print("could not read shifts from config; pass --shifts D1_snr,D2_cfo")
        return 2
    sf = [int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds else None
    m = Model(root, args.backbones.split(","), shifts, seed_filter=sf)

    if args.once:
        print(render(m, not args.no_log, None))
        return 0

    clear = (not args.no_clear) and sys.stdout.isatty()
    try:
        while True:
            text = render(m, not args.no_log, None)
            if clear:
                sys.stdout.write(CLEAR + text + "\n")
            else:
                sys.stdout.write(text + "\n" + "-" * W + "\n")
            sys.stdout.flush()
            don, tot = m.counts()
            if all(don[k] == tot[k] for k in tot) and sum(tot.values()) > 0 and not m.running():
                if clear:
                    sys.stdout.write(CLEAR + render(m, False, None) + "\n")
                    sys.stdout.flush()
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nexited (experiments keep running)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
