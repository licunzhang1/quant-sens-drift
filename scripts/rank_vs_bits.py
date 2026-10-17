"""How rho depends on the **reference bit width**.

Why this is needed
------------------
The per-layer sensitivity is measured at some "reference bit width" b*. A badly
chosen b* ruins the rank analysis:

* b* too large (e.g. 8 bit) -> the Delta loss of nearly every layer is ~= 0 ->
  **many tied values** -> the rank correlation is decided by tiny, essentially
  random differences;
* b* too small (e.g. 2 bit) -> every layer degrades badly -> the differences get
  compressed and are equally hard to separate;
* only some intermediate rung is likely to give a ranking with real resolution.

Measured (CLDNN, D4_phasenoise): at reference bit width 4 bit, rho=+0.139
(p=0.71), yet the domain gap is actually large (acc 0.965 -> 0.540). So a "low
rho" may just mean the bit width was chosen badly, rather than that "the ranking
really does not transfer". **These two explanations lead to opposite
conclusions** and must be told apart.

Usage
-----
    python scripts/rank_vs_bits.py                      # scan every result under results/sensitivity
    python scripts/rank_vs_bits.py --shift D4_phasenoise
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
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.rank_corr import analyse_pair  # noqa: E402
from src.utils import discover_seed_roots  # noqa: E402
from src.utils import dump_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=None,
                    help="only look at one backbone; defaults to all")
    ap.add_argument("--shift", default=None, help="only look at one shift; defaults to all")
    ap.add_argument("--n-permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=None,
                    help="which seed's results to analyse; default: the highest available")
    ap.add_argument("--outdir", default="results/rank_vs_bits")
    args = ap.parse_args()

    results_dir = ROOT / "results"
    seed = args.seed
    if seed is None:
        roots = discover_seed_roots(results_dir)
        if not roots:
            print(f"no results under {results_dir}; run scripts/run_sensitivity.py first")
            return 2
        seed = max(roots)
    sens_dir = results_dir / f"seed{seed}" / "sensitivity"
    if not sens_dir.is_dir():
        sens_dir = results_dir / "sensitivity"   # flat fallback
    files = sorted(sens_dir.glob("sens_*.npz"))
    if not files:
        print(f"no sens_*.npz under {sens_dir}; run scripts/run_sensitivity.py first")
        return 2
    print(f"analysing seed {seed}: {sens_dir.relative_to(ROOT)}")

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
        print(f"  {backbone} x {shift}   layers n={len(layer_names)}   "
              f"bit ladder {ladder}")
        print(f"{'=' * 78}")
        print(f"  {'ref_bits':>8} | {'rho_loss':>9} {'p_loss':>8} "
              f"{'#uniq_A':>8} {'#uniq_B':>8} | {'rho_acc':>8} {'p_acc':>8}")
        print("  " + "-" * 72)

        for j, b in enumerate(ladder):
            sa, sb = dl_a[:, j], dl_b[:, j]
            ta, tb = da_a[:, j], da_b[:, j]

            na, nb = int(np.unique(sa).size), int(np.unique(sb).size)
            ta_u, tb_u = int(np.unique(ta).size), int(np.unique(tb).size)

            # `seed` here is the RNG seed for the permutation test, which is
            # a different thing from the seed directory being analysed.
            rng_seed = seed if isinstance(seed, int) else 42
            rl = analyse_pair(sa, sb, n_permutations=args.n_permutations,
                              seed=rng_seed, n_boot=0)
            ra = analyse_pair(ta, tb, n_permutations=args.n_permutations,
                              seed=rng_seed, n_boot=0)

            # too many ties makes rho unreliable; flag it
            tie_flag = "  <-- too many ties" if (na < len(layer_names) * 0.8
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

        # report the most trustworthy reference bit width for this combo:
        # fewest ties and the largest effect
        sub = [r for r in rows if r["backbone"] == backbone and r["shift"] == shift]
        if sub:
            best = max(sub, key=lambda r: (min(r["n_unique_loss_A"], r["n_unique_loss_B"]),
                                           abs(r["rho_loss"])))
            print(f"\n  reference bit width with fewest ties = {best['ref_bits']} bit  "
                  f"(unique A={best['n_unique_loss_A']}, B={best['n_unique_loss_B']}, "
                  f"rho={best['rho_loss']:+.4f}, p={best['p_loss']:.4f})")

    if not rows:
        print("no matching results.")
        return 1

    # ---------------------------------------------------------------- summary
    print(f"\n{'=' * 78}")
    print("  summary: rho(loss) for each (backbone, shift) at every reference bit width")
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
    print("\n  * marks p < 0.05. A blank means that bit width is not in this "
          "combo's ladder.")

    out = dump_json(ROOT / args.outdir / "rank_vs_bits.json", {
        "n_permutations": args.n_permutations, "seed": args.seed, "rows": rows,
    })
    print(f"\n  wrote {out}")
    print("\n  [!] How to read this: if different reference bit widths give very"
          " different rho values (say +0.14 at 4b but +0.7 at 2b),")
    print("    then **the reference bit width is itself a design choice that needs"
          " to be justified**,")
    print("    and the report must say why it was chosen and include a sensitivity"
          " analysis over the bit width.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
