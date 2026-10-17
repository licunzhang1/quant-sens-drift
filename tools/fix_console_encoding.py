"""Fix GBK console encoding problems.

Background
----------
A Chinese Windows console defaults to GBK. Any character that cannot be encoded
to GBK raises UnicodeEncodeError on print and crashes outright -- measured:
U+26A0 makes scripts/*.py and tests/smoke_pipeline.py fail in a fresh clone.

Two-step fix
------------
1. **Bootstrap**: after the module docstring of every executable script, insert
   a stdout/stderr reconfiguration that sets errors to replace. Then even if
   someone adds a new special character later, the worst case is that it shows
   up as "?" instead of crashing.
2. **Cleanup**: replace the 15 non-encodable characters currently in use with
   their ASCII equivalents.

Locating the end of the docstring with ast is more reliable than a regex.

Usage
-----
    python tools/fix_console_encoding.py --dry-run
    python tools/fix_console_encoding.py
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# character -> ASCII replacement table
REPLACEMENTS = {
    "\u2212": "-",        # MINUS SIGN
    "\u26a0": "[!]",      # WARNING SIGN
    "\u2113": "l",        # SCRIPT SMALL L (layer index)
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
    """Return ``(path, whether the console bootstrap needs inserting)``.

    * executable scripts in ``scripts/`` ``tests/`` ``tools/``: need the
      bootstrap (they print)
    * library modules in ``src/``: character replacement only (a library must
      not touch the caller's stdout)
    * ``.md`` / ``.yaml`` / ``.txt`` / ``.cff``: character replacement only
      (clean up typographic characters)
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
    """Return the end line number of the module docstring (1-based); 0 if absent."""
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
    """Insert the bootstrap after the module docstring (and any ``from __future__``
    block).

    Key constraint: ``from __future__ import ...`` must be the **first statement**
    after the docstring, or the whole file is a SyntaxError. The insertion point
    must therefore skip it, along with any blank lines in between.
    """
    lines = text.splitlines(keepends=True)
    at = docstring_end_line(text)

    # skip blank lines and the from __future__ line -- neither may end up
    # after the bootstrap code
    j = at
    while j < len(lines):
        s = lines[j].strip()
        if s == "" or lines[j].startswith("from __future__"):
            j += 1
            continue
        break
    at = j

    if at == 0:
        # no docstring and no __future__: insert before the first import
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
