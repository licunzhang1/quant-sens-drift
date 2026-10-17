"""W15-W16: generate the four core report figures from saved results.

    python scripts/make_figures.py --backbone cldnn --shift D1_snr

Reads:
  results/sensitivity/sens_<backbone>_<shift>.npz      <- run_sensitivity.py
  results/equal_error/arm_<backbone>_<shift>.json      <- run_equal_error.py
  results/baseline/<backbone>_metrics.json             <- train_baseline.py (optional)

Writes:
  results/figures/fig1_sensitivity_heatmap.png
  results/figures/fig2_rank_scatter.png
  results/figures/fig3_equal_error_arms.png
  results/figures/fig4_proxy_scatter.png
  results/figures/fig5_domain_gap.png            (when baseline metrics exist)

**This script is the single authoritative entry point for these figures.**
Do not draw figures by hand -- every figure in the report must be reproducible
from a single command, which is part of reproducibility.
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

from src.utils import discover_seed_roots  # noqa: E402
from src.viz.figures import (  # noqa: E402
    plot_domain_gap,
    plot_equal_error_arms,
    plot_proxy_scatter,
    plot_rank_scatter,
    plot_sensitivity_heatmap,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="cldnn")
    ap.add_argument("--shift", default="D1_snr")
    ap.add_argument("--reference-bits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None,
                    help="which seed's results to plot; default: the highest available")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    tag = f"{args.backbone}_{args.shift}"
    results_dir = pathlib.Path("results")
    seed = args.seed
    if seed is None:
        roots = discover_seed_roots(results_dir)
        seed = max(roots) if roots else None
    seed_dir = (results_dir / f"seed{seed}") if seed is not None else results_dir
    if not (seed_dir / "sensitivity").is_dir():
        seed_dir = results_dir                     # flat fallback
    print(f"reading from {seed_dir}")
    sens_npz = seed_dir / "sensitivity" / f"sens_{tag}.npz"
    arm_json = seed_dir / "equal_error" / f"arm_{tag}.json"
    base_json = seed_dir / "baseline" / f"{args.backbone}_metrics.json"
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not sens_npz.exists():
        print(f"missing {sens_npz}; run scripts/run_sensitivity.py first")
        return 2
    z = np.load(sens_npz, allow_pickle=True)
    layer_names = [str(s) for s in z["layer_names"]]
    ladder = [int(b) for b in z["bits_ladder"]]
    ref = int(z["reference_bits"][0]) if "reference_bits" in z else args.reference_bits
    idx = ladder.index(ref)
    dl_a, dl_b = z["delta_loss_a"], z["delta_loss_b"]
    da_a, da_b = z["delta_acc_a"], z["delta_acc_b"]

    made = []
    made.append(plot_sensitivity_heatmap(
        dl_a, dl_b, layer_names, ladder,
        outdir / "fig1_sensitivity_heatmap.png",
        name_a=f"{args.backbone} / Domain A (loss)", 
        name_b=f"{args.backbone} / {args.shift} (loss)"))

    # rho and p are read from rank_*.json (annotated only, not recomputed)
    rank_json = seed_dir / "sensitivity" / f"rank_{tag}.json"
    rho = p = float("nan")
    if rank_json.exists():
        rk = json.loads(rank_json.read_text(encoding="utf-8"))
        row = rk.get("rank_correlation", {}).get("loss", {})
        rho = row.get("rho", float("nan"))
        p = row.get("p_value", float("nan"))
    made.append(plot_rank_scatter(
        dl_a[:, idx], dl_b[:, idx], layer_names, rho, p,
        outdir / "fig2_rank_scatter.png", reference_bits=ref))

    if arm_json.exists():
        arm = json.loads(arm_json.read_text(encoding="utf-8"))
        arms_all = arm.get("arms", [])
        if arms_all:
            # take the middle budget rung (three rungs by default: 0.4/0.6/0.8)
            pick = sorted(arms_all, key=lambda r: abs(r["budget_ratio"] - 0.6))[0]
            made.append(plot_equal_error_arms(
                {"uniform": pick["uniform_bits"],
                 "agnostic": pick["agnostic_bits"],
                 "oracle": pick["oracle_bits"]},
                {"uniform": pick["uniform_damage"],
                 "agnostic": pick["agnostic_damage"],
                 "oracle": pick["oracle_damage"]},
                ladder, layer_names, outdir / "fig3_equal_error_arms.png",
                budget_ratio=pick["budget_ratio"]))
        proxies = arm.get("proxies", [])
        if proxies:
            # the raw proxy values are not stored in the arm json, so the x axis
            # shows a stand-in built from the sensitivities
            made.append(plot_proxy_scatter(
                {c["proxy"]: np.array([c["rho"]]) for c in proxies},
                np.array([0.0]), proxies, outdir / "fig4_proxy_correlation.png"))
            print("  note: fig4 plots an |rho| summary of the proxies;"
                  " to draw a real scatter plot, the raw proxy values would have to"
                  " be saved by run_equal_error.py as well.")
    else:
        print(f"missing {arm_json}; skipping fig3/fig4")

    if base_json.exists():
        b = json.loads(base_json.read_text(encoding="utf-8"))
        doms = b.get("domains", {})
        if doms:
            made.append(plot_domain_gap(
                {k: v["acc"] for k, v in doms.items()},
                outdir / "fig5_domain_gap.png", metric="accuracy"))

    print("\nfigures written:")
    for p_ in made:
        print(f"  {p_}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
