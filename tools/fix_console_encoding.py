"""修复 GBK 控制台编码问题。

背景
----
中文 Windows 的控制台默认编码是 GBK。任何无法编码到 GBK 的字符在 print 时
会抛 UnicodeEncodeError 直接崩溃 —— 实测 U+26A0 会让 scripts/*.py 和
tests/smoke_pipeline.py 在克隆副本里失败。

两步修复
--------
1. **兜底**：在每个可执行脚本的模块 docstring 之后插入一段 stdout/stderr
   重配置，把 errors 设为 replace。这样将来即使有人加入新的特殊字符，
   最差也只是显示成 "?" 而不会崩溃。
2. **清理**：把当前用到的 15 种不可编码字符替换为 ASCII 等价物。

用 ast 定位 docstring 的结束位置，比正则可靠。

用法
----
    python tools/fix_console_encoding.py --dry-run
    python tools/fix_console_encoding.py
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 字符 -> ASCII 替换表
REPLACEMENTS = {
    "\u2212": "-",        # MINUS SIGN
    "\u26a0": "[!]",      # WARNING SIGN
    "\u2113": "l",        # SCRIPT SMALL L（层索引）
    "\u00b2": "^2",       # SUPERSCRIPT TWO
    "\u00b3": "^3",       # SUPERSCRIPT THREE
    "\u2074": "^4",       # SUPERSCRIPT FOUR
    "\u2076": "^6",       # SUPERSCRIPT SIX
    "\u2705": "[done]",   # WHITE HEAVY CHECK MARK
    "\u2713": "OK",       # CHECK MARK
    "\u2717": "X",        # BALLOT X
    "\u2b50": "",         # WHITE MEDIUM STAR
    "\u30fb": "-",        # KATAKANA MIDDLE DOT
    "\u274c": "[X]",      # CROSS MARK
    "\U0001f4cc": "",     # PUSHPIN
    "\u00b9": "^1",       # SUPERSCRIPT ONE
}

MARKER = "reconfigure(errors="

BOOTSTRAP = '''# --- console safety --------------------------------------------------------
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
'''


def target_files() -> list[tuple[pathlib.Path, bool]]:
    """返回 ``(路径, 是否需要插入控制台兜底代码)``。

    * ``scripts/`` ``tests/`` ``tools/`` 里的可执行脚本：需要兜底（它们会 print）
    * ``src/`` 里的库模块：只做字符替换（库不应改调用方的 stdout）
    * ``.md`` / ``.yaml`` / ``.txt`` / ``.cff``：只做字符替换（清理排版字符）
    """
    self_path = pathlib.Path(__file__).resolve()
    out: list[tuple[pathlib.Path, bool]] = []
    for sub in ("scripts", "tests", "tools"):
        d = ROOT / sub
        if d.is_dir():
            out += [(p, True) for p in sorted(d.glob("*.py")) if p.resolve() != self_path]
    src = ROOT / "src"
    if src.is_dir():
        out += [(p, False) for p in sorted(src.rglob("*.py"))]
    for pat in ("*.md", "*.yaml", "*.txt", "*.cff"):
        out += [(p, False) for p in sorted(ROOT.rglob(pat))
                if ".git" not in p.parts and "__pycache__" not in p.parts]
    return out


def docstring_end_line(text: str) -> int:
    """返回模块 docstring 的结束行号（1-based）；没有则返回 0。"""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0
    if not tree.body:
        return 0
    first = tree.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return int(first.end_lineno or 0)
    return 0


def insert_bootstrap(text: str) -> str:
    """在模块 docstring（以及其后的 ``from __future__`` 块）之后插入兜底代码。

    关键约束：``from __future__ import ...`` 必须是 docstring 之后的**第一条语句**，
    否则整个文件是 SyntaxError。因此插入点必须跳过它，以及中间的空行。
    """
    lines = text.splitlines(keepends=True)
    at = docstring_end_line(text)

    # 跳过空行与 from __future__ 行 —— 两者都不能被兜底代码插到前面
    j = at
    while j < len(lines):
        s = lines[j].strip()
        if s == "" or lines[j].startswith("from __future__"):
            j += 1
            continue
        break
    at = j

    if at == 0:
        # 没有 docstring 也没有 __future__：插到第一个 import 之前
        for i, ln in enumerate(lines):
            if ln.startswith(("import ", "from ")):
                at = i
                break

    head = "".join(lines[:at])
    tail = "".join(lines[at:])
    if head and not head.endswith("\n\n"):
        head = head.rstrip("\n") + "\n\n"
    return head + BOOTSTRAP + "\n" + tail


def fix_file(p: pathlib.Path, add_bootstrap: bool, dry: bool) -> tuple[bool, int, bool]:
    original = p.read_text(encoding="utf-8")
    text = original
    n_repl = 0
    for bad, good in REPLACEMENTS.items():
        if bad in text:
            n_repl += text.count(bad)
            text = text.replace(bad, good)
    added = False
    if add_bootstrap and MARKER not in text:
        text = insert_bootstrap(text)
        added = True
    changed = text != original
    if changed and not dry:
        p.write_text(text, encoding="utf-8")
    return changed, n_repl, added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = target_files()
    print(f"{'DRY RUN' if args.dry_run else 'APPLY'}  target files: {len(targets)}")
    n_files = n_repl = n_boot = 0
    for p, add_boot in targets:
        changed, nr, added = fix_file(p, add_boot, args.dry_run)
        if changed:
            n_files += 1
            n_repl += nr
            n_boot += int(added)
            tag = "bootstrap added" if added else ("scan-only" if not add_boot else "bootstrap present")
            print(f"  {p.relative_to(ROOT)}  replacements={nr}  {tag}")
    print()
    print(f"  files changed          : {n_files}")
    print(f"  character replacements : {n_repl}")
    print(f"  bootstraps added       : {n_boot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
