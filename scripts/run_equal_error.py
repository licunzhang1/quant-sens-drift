"""W11-W12: equal reconstruction error arms -- a three-arm comparison.

    python scripts/run_equal_error.py --config configs/base.yaml --shift D1_snr

Reads ``results/sensitivity/sens_<backbone>_<shift>.npz`` (produced by
run_sensitivity.py) plus the backbone weights, computes the per-layer
reconstruction error curve eps_l(b), and runs three arms:

    uniform   the same bit width for every layer
    agnostic  allocate by **source-domain** sensitivity
              <- what you can actually do (you only know the training domain)
    oracle    allocate by **target-domain** sensitivity
              <- not realizable; it is the upper bound

Two key differences:
    agnostic - uniform   does sensitivity guidance help at all
    oracle  - agnostic    **the cost of not knowing the deployment domain**
                         <- the core quantity of this project

If the latter is not negligible we get a clean counterexample: **no
domain-independent optimal bit allocation exists**.
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
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import build_model, load_config  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.equal_error import run_equal_error_arm  # noqa: E402
from src.analysis.proxies import correlate_with_sensitivity, proxy_matrix  # noqa: E402
from src.quant.core import quant_error_curve  # noqa: E402
from src.utils import dump_json, find_result_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--shift", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--sens", default=None, help="path to a sens_*.npz file")
    ap.add_argument("--ckpt", default=None,
                    help="backbone checkpoint path (default results/baseline/<name>.pt)")
    ap.add_argument("--outdir", default="results/equal_error")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the seed from the config (used by the multi-seed runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.backbone:
        cfg["model"]["name"] = args.backbone
    name = cfg["model"]["name"]
    shifts = list((cfg["data"].get("shifts") or {}).keys())
    shift = args.shift or (shifts[0] if shifts else None)

    sens_path = (pathlib.Path(args.sens) if args.sens
                 else find_result_file(ROOT / "results", "sensitivity",
                                       f"sens_{name}_{shift}.npz"))
    if sens_path is None:
        print(f"no sens_{name}_{shift}.npz found under results/; "
              "run scripts/run_sensitivity.py first")
        return 2
    if not sens_path.exists():
        print(f"{sens_path} not found; run scripts/run_sensitivity.py first")
        return 2
    z = np.load(sens_path, allow_pickle=True)
    layer_names = [str(s) for s in z["layer_names"]]
    ladder = [int(b) for b in z["bits_ladder"]]
    ref_bits = int(z["reference_bits"][0]) if "reference_bits" in z else int(
        cfg["quant"].get("reference_bits", 4))
    delta_loss_a, delta_loss_b = z["delta_loss_a"], z["delta_loss_b"]
    idx = ladder.index(ref_bits)
    s_src = delta_loss_a[:, idx]
    s_tgt = delta_loss_b[:, idx]

    print(f"[run_equal_error] {name}  target={shift}  layers={len(layer_names)}  "
          f"reference_bits={ref_bits}")

    # ---------------------------------------------------------------- error curves
    ckpt = pathlib.Path(args.ckpt or f"results/baseline/{name}.pt")
    if not ckpt.exists():
        print(f"checkpoint {ckpt} not found")
        return 2
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg, n_classes=int(blob.get("n_classes", 4)))
    model.load_state_dict(blob["state_dict"])

    per_channel = bool(cfg["quant"].get("per_channel_weights", True))
    axis = 0 if per_channel else None
    err = np.zeros((len(layer_names), len(ladder)))
    costs = np.zeros(len(layer_names))
    tensors = {}
    modules = dict(model.named_modules())
    for i, nm in enumerate(layer_names):
        if nm not in modules or getattr(modules[nm], "weight", None) is None:
            raise KeyError(f"no quantizable weight layer {nm!r} in the model")
        W = modules[nm].weight.detach().cpu().numpy().astype(np.float64)
        tensors[nm] = W
        costs[i] = W.size
        curve = quant_error_curve(W, ladder, symmetric=True, per_channel_axis=axis)
        for j, b in enumerate(ladder):
            err[i, j] = curve[b]["nmse"]

    print("\n  per-layer NMSE eps_l(b)")
    print("    {:<34s}".format("layer") + "".join(f"{b:>10d}b" for b in ladder))
    for i, nm in enumerate(layer_names):
        print("    {:<34s}".format(nm[:32])
              + "".join(f"{err[i, j]:>11.5f}" for j in range(len(ladder))))

    # ---------------------------------------------------------------- three arms
    ecfg = cfg.get("equal_error", {})
    ratio = float(ecfg.get("total_bit_budget", 0.8))
    w_by_params = bool(ecfg.get("weight_by_params", True))
    costs_use = costs if w_by_params else np.ones_like(costs)

    print(f"\n  budget ratio={ratio}  weighted by param count={w_by_params}")
    rows_all = []
    for r in (0.4, 0.6, 0.8):
        arm = run_equal_error_arm(s_src, s_tgt, err, ladder, budget_ratio=r,
                                  layer_costs=costs_use, layer_names=layer_names,
                                  source_domain="A", target_domain=shift)
        print(f"\n  --- budget {r} ---")
        print("    {:<20s}{:<30s}{:>16s}".format("arm", "bits", "pred_damage"))
        for tag2, a in (("uniform", arm.uniform), ("agnostic(source)", arm.agnostic),
                        ("oracle(target)", arm.oracle)):
            print("    {:<20s}{:<30s}{:>16.8f}".format(
                tag2, str(a.bits.tolist()), a.predicted_damage))
        print(f"    sensitivity gain (uniform-agnostic) = {arm.gain_from_sensitivity:+.8f}")
        print(f"    **cost of domain ignorance** (agnostic-oracle) "
              f"= {arm.cost_of_domain_ignorance:+.8f}"
              f"   relative {arm.relative_cost_of_ignorance:+.1%}")
        print(f"    ranking flips the decision: {arm.ranking_flipped}")
        print(f"    verdict: {arm.verdict()}")
        rows_all.append({"budget_ratio": r, **{
            k: v for k, v in {
                "uniform_bits": arm.uniform.bits, "agnostic_bits": arm.agnostic.bits,
                "oracle_bits": arm.oracle.bits,
                "uniform_damage": arm.uniform.predicted_damage,
                "agnostic_damage": arm.agnostic.predicted_damage,
                "oracle_damage": arm.oracle.predicted_damage,
                "gain_from_sensitivity": arm.gain_from_sensitivity,
                "cost_of_domain_ignorance": arm.cost_of_domain_ignorance,
                "relative_cost_of_ignorance": arm.relative_cost_of_ignorance,
                "ranking_flipped": arm.ranking_flipped, "verdict": arm.verdict(),
            }.items()}})

    # ---------------------------------------------------------------- proxies
    print("\n  proxies vs target-domain sensitivity (|rho| descending)")
    prox, _ = proxy_matrix(tensors, bits=ref_bits)
    n_perm = int(cfg.get("rank", {}).get("n_permutations", 10000))
    pres = correlate_with_sensitivity(prox, s_tgt, n_permutations=n_perm,
                                      seed=int(cfg.get("seed", 42)))
    for r in pres:
        print(f"    {r['proxy']:<14s} rho={r['rho']:+.4f}  p={r['p_value']:.4f}"
              + ("  *" if r["significant"] else ""))

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = dump_json(outdir / f"arm_{name}_{shift}.json", {
        "backbone": name, "target_domain": shift, "reference_bits": ref_bits,
        "layer_names": layer_names, "bits_ladder": ladder,
        "source_sensitivity": s_src, "target_sensitivity": s_tgt,
        "err_curves": err, "layer_costs": costs, "weighted_by_params": w_by_params,
        "arms": rows_all, "proxies": pres,
    })
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
