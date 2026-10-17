#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Replace the remaining non-ASCII characters under src/ with ASCII.

Why
---
`src/` was translated to English but kept mathematical glyphs (rho, Delta,
arrow, in, degree, phi). Those are fine on GitHub but garble on a console
whose encoding cannot represent them -- and most are also unreadable in a
plain-text diff. Since the rest of the repository is now pure ASCII, make
`src/` consistent.

Every replacement is a semantic no-op: `rho` means the same as the Greek
letter, `->` the same as the arrow. No number, identifier or format spec
changes.

Usage
-----
    python tools/ascii_ify_src.py --dry-run
    python tools/ascii_ify_src.py
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

# --- console safety --------------------------------------------------------
import sys as _console_sys

for _stream in (_console_sys.stdout, _console_sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass
del _console_sys, _stream
# --- end console safety ----------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]

REPLACEMENTS = {
    "\u03c1": "rho",       # greek small letter rho
    "\u0394": "Delta",     # greek capital letter delta
    "\u03a3": "Sum",       # greek capital letter sigma
    "\u03b5": "eps",       # greek small letter epsilon
    "\u03c4": "tau",       # greek small letter tau
    "\u03b1": "alpha",     # greek small letter alpha
    "\u03c0": "pi",        # greek small letter pi
    "\u03c6": "phi",       # greek small letter phi
    "\u2208": " in ",      # element of
    "\u00b0": " deg",      # degree sign
    "\u2192": "->",        # rightwards arrow
    "\u2190": "<-",        # leftwards arrow
    "\u2248": "~=",        # almost equal to
    "\u2261": "==",        # identical to
    "\u2265": ">=",        # greater-than or equal to
    "\u2264": "<=",        # less-than or equal to
    "\u00d7": "x",         # multiplication sign
    "\u00b7": "*",         # middle dot
    "\u2212": "-",         # minus sign
    "\u00b1": "+/-",       # plus-minus sign
    "\u221e": "inf",       # infinity
    "\u221a": "sqrt",      # square root
    "\u2113": "l",         # script small l
    "\u00b2": "^2",        # superscript two
    "\u00b3": "^3",        # superscript three
    "\u2074": "^4",        # superscript four
    "\u2076": "^6",        # superscript six
    "\u00b9": "^1",        # superscript one
    "\u2264": "<=",
    "\u2013": "-",         # en dash
    "\u2014": " -- ",      # em dash
    "\u2018": "'",         # left single quote
    "\u2019": "'",         # right single quote
    "\u201c": '"',         # left double quote
    "\u201d": '"',         # right double quote
    "\u2026": "...",       # horizontal ellipsis
}


def main() -> int:
    dry = "--dry-run" in sys.argv
    src = ROOT / "src"
    files = sorted(src.rglob("*.py"))
    print(f"{'DRY RUN' if dry else 'APPLY'}  files: {len(files)}")

    counter: Counter = Counter()
    changed = 0
    for p in files:
        text = p.read_text(encoding="utf-8")
        original = text
        for bad, good in REPLACEMENTS.items():
            if bad in text:
                counter[bad] += text.count(bad)
                text = text.replace(bad, good)
        if text != original:
            changed += 1
            if not dry:
                p.write_text(text, encoding="utf-8")
            print(f"  {p.relative_to(ROOT)}")

    print()
    print(f"  files changed: {changed}")
    print("  replacements:")
    for ch, n in counter.most_common():
        print(f"    {ascii(ch):<10} {REPLACEMENTS[ch]!r:<10} x{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
