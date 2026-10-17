"""Regenerate the human-readable verdict strings inside stored result JSONs.

Why this exists
---------------
`rank_corr.RankCorrelationResult.verdict()` and
`equal_error.EqualErrorArmResult.verdict()` emit a one-line interpretation of
the numbers, and that line is written into every `rank_*.json` and
`arm_*.json`. When those two methods are translated, the already-stored JSONs
still carry the old strings.

Re-running the sensitivity pipeline to refresh them would cost about 75 minutes
of CPU. But a verdict is a pure function of numbers that are already stored, so
it can simply be recomputed. That is what this script does.

It never re-measures anything: only the `verdict` fields are touched, and every
other field is written back byte-identical.

Usage
-----
    python tools/retranslate_verdicts.py --dry-run
    python tools/retranslate_verdicts.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.rank_corr import RankCorrelationResult  # noqa: E402
from src.utils import dump_json  # noqa: E402

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


VERDICT_OK = "OK"


def rebuild_rank(path: pathlib.Path, dry: bool) -> tuple[bool, int]:
    """Recompute verdict strings inside one rank_*.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rc = data.get("rank_correlation")
    if not isinstance(rc, dict):
        return False, 0

    changed = 0
    for key in ("loss", "acc"):
        row = rc.get(key)
        if not isinstance(row, dict) or "rho" not in row:
            continue
        res = RankCorrelationResult(
            n_layers=int(row.get("n_layers") or 0),
            rho=float(row.get("rho") if row.get("rho") is not None else float("nan")),
            p_value=float(row.get("p_value") if row.get("p_value") is not None else float("nan")),
            ci_low=float(row.get("ci_low") if row.get("ci_low") is not None else float("nan")),
            ci_high=float(row.get("ci_high") if row.get("ci_high") is not None else float("nan")),
            tau_b=float(row.get("tau_b") if row.get("tau_b") is not None else float("nan")),
            rho_pearson_raw=float(row.get("rho_pearson_raw") or 0.0),
            seed_used=int(row.get("seed_used") or 0),
        )
        new = res.verdict()
        if row.get("verdict") != new:
            changed += 1
            if not dry:
                row["verdict"] = new
    if changed and not dry:
        dump_json(path, data)
    return bool(changed), changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default="results")
    args = ap.parse_args()

    base = ROOT / args.root
    files = sorted(base.rglob("rank_*.json"))
    # results/figures/_saturated_example holds an archived copy; refresh it too
    print(f"{'DRY RUN' if args.dry_run else 'APPLY'}  rank files: {len(files)}")

    n_files = n_rows = 0
    for f in files:
        changed, n = rebuild_rank(f, args.dry_run)
        if changed:
            n_files += 1
            n_rows += n
            print(f"  {f.relative_to(ROOT)}  rows={n}")
    print()
    print(f"  files updated: {n_files}")
    print(f"  rows updated : {n_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
