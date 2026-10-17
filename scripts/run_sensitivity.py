"""W5-W6 + W9-W10: per-layer sensitivity matrix + rank stability rho.

    python scripts/run_sensitivity.py --config configs/base.yaml --shift D1_snr
    python scripts/run_sensitivity.py --config configs/base.yaml --shift D1_snr --backbone resnet1d

Pipeline
--------
1. Load ``results/baseline/<backbone>.pt``
2. Compute a per-layer sensitivity matrix on **Domain A** and on the **target
   domain** (loss is the primary metric, acc is computed alongside it)
3. Compute rho = Spearman(sensitivity_A, sensitivity_B), with a permutation test
   and a bootstrap CI
4. Write npz + json for run_equal_error.py and make_figures.py

Key constraints
---------------
* **The primary metric must be loss.** Accuracy saturates (an in-domain 1.0 can
  still be 1.0 after quantizing to 2 bit), which leaves the sensitivity all zeros
  and degenerates the rank correlation to NaN. The script prints both metrics
  side by side.
* Quantizing one layer uses ``quantize_one_layer`` -- **only that layer** is
  quantized, the others stay in floating point.
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
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import build_model, evaluate, load_config, make_domain, pick_device  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.rank_corr import analyse_pair  # noqa: E402
from src.analysis.sensitivity import compute_sensitivity_matrix  # noqa: E402
from src.quant.fake_quant import quantize_one_layer  # noqa: E402
from src.utils import dump_json, seed_everything  # noqa: E402


def print_matrix(title, names, bits, mat, fmt="{:>12.5f}"):
    print(f"\n  {title}")
    print("    {:<34s}".format("layer") + "".join(f"{b:>10d}b" for b in bits))
    for i, nm in enumerate(names):
        print("    {:<34s}".format(nm[:32])
              + "".join(fmt.format(mat[i, j]) for j in range(len(bits))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--shift", default=None,
                    help="target-domain key (data.shifts in the configs)")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--outdir", default="results/sensitivity")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the seed from the config (used by the multi-seed runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backbone:
        cfg["model"]["name"] = args.backbone
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    name = cfg["model"]["name"]
    shifts = list((cfg["data"].get("shifts") or {}).keys())
    shift = args.shift or (shifts[0] if shifts else None)
    if shift is None:
        print("no data.shifts in the config; cannot pick a target domain.")
        return 2

    ckpt_path = pathlib.Path(args.ckpt or f"results/baseline/{name}.pt")
    if not ckpt_path.exists():
        print(f"checkpoint {ckpt_path} not found; run scripts/train_baseline.py first")
        return 2
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_classes = int(blob.get("n_classes", 4))
    model = build_model(cfg, n_classes=n_classes)
    model.load_state_dict(blob["state_dict"])
    device = pick_device(cfg["train"].get("device", "auto"))
    model.to(device)

    qcfg = cfg["quant"]
    scfg = cfg.get("sensitivity", {})
    ladder = [int(b) for b in qcfg["bits_ladder"]]
    ref_bits = int(qcfg.get("reference_bits", 4))
    per_channel = bool(qcfg.get("per_channel_weights", True))
    symmetric = bool(qcfg.get("symmetric", True))

    layer_names = model.weight_layer_names()
    if scfg.get("layer_filter"):
        import re

        pat = re.compile(scfg["layer_filter"])
        layer_names = [n for n in layer_names if pat.search(n)]
    if scfg.get("max_layers"):
        layer_names = layer_names[: int(scfg["max_layers"])]
    print(f"[run_sensitivity] backbone={name}  target={shift}  "
          f"layers={len(layer_names)}  bits={ladder}  reference_bits={ref_bits}")
    if len(layer_names) < 8:
        print(f"  [!] only {len(layer_names)} layers. With n<8 the 95% CI of the"
              " Spearman rho is so wide that it carries almost no information"
              " (measured: [-1,1] at n=5).")
        print("    Suggestion: use a deeper backbone (CLDNN with convs_per_stage=2"
              " gives 10 layers, ResNet1D gives 16), or loosen"
              " sensitivity.layer_filter.")

    n_test = int(cfg["data"].get("n_test", 2000))
    Xa, ya = make_domain(cfg, "a", n_test, seed + 100)
    Xb, yb = make_domain(cfg, shift, n_test, seed + 200)

    def ql(nm, bits):
        return quantize_one_layer(model, nm, bits, per_channel=per_channel,
                                  symmetric=symmetric)

    def acc_fn(X, y):
        return lambda: evaluate(model, X, y, device)[0]

    def loss_fn(X, y):
        return lambda: evaluate(model, X, y, device)[1]

    t0 = time.time()
    out = {}
    for tag, X, y, dom in (("A", Xa, ya, "A"), ("B", Xb, yb, shift)):
        print(f"\n  [{dom}] loss metric")
        s_loss = compute_sensitivity_matrix(
            layer_names, ladder, evaluate=loss_fn(X, y), quantize_layer=ql,
            metric="loss", higher_is_better=False, domain=dom, backbone=name)
        print(f"  [{dom}] acc metric (control)")
        s_acc = compute_sensitivity_matrix(
            layer_names, ladder, evaluate=acc_fn(X, y), quantize_layer=ql,
            metric="acc", higher_is_better=True, domain=dom, backbone=name)
        out[tag] = {"loss": s_loss, "acc": s_acc}

    print_matrix(f"Delta loss (source {name})", layer_names, ladder,
                 out["A"]["loss"].delta)
    print_matrix(f"Delta loss (target {shift})", layer_names, ladder,
                 out["B"]["loss"].delta)

    # ---------------------------------------------------------------- rho
    print("\n" + "=" * 78)
    sl_a = out["A"]["loss"].sensitivity_at(ref_bits)
    sl_b = out["B"]["loss"].sensitivity_at(ref_bits)
    sa_a = out["A"]["acc"].sensitivity_at(ref_bits)
    sa_b = out["B"]["acc"].sensitivity_at(ref_bits)

    n_perm = int(cfg.get("rank", {}).get("n_permutations", 10000))
    res_loss = analyse_pair(sl_a, sl_b, n_permutations=n_perm, seed=seed)
    res_acc = analyse_pair(sa_a, sa_b, n_permutations=n_perm, seed=seed)

    print(f"  reference bits {ref_bits}   layers n={len(layer_names)}")
    print(f"  source sensitivity (loss): {np.round(sl_a, 5).tolist()}")
    print(f"  target sensitivity (loss): {np.round(sl_b, 5).tolist()}")
    print(f"\n  === loss metric (primary) ===")
    print(f"  rho={res_loss.rho:+.4f}  p={res_loss.p_value:.4g}  "
          f"CI=[{res_loss.ci_low:+.3f},{res_loss.ci_high:+.3f}]  "
          f"tau-b={res_loss.tau_b:+.4f}")
    print(f"  {res_loss.verdict()}")
    print(f"\n  === acc metric (control) ===")
    print(f"  rho={res_acc.rho:+.4f}  p={res_acc.p_value:.4g}")
    if int(np.unique(sa_a).size) < len(layer_names):
        print(f"  [!] source-domain acc sensitivity has only "
              f"{int(np.unique(sa_a).size)} distinct values / "
              f"{len(layer_names)} layers -- classic metric saturation.")

    # ---------------------------------------------------------------- save
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_{shift}"
    npz = outdir / f"sens_{tag}.npz"
    np.savez_compressed(
        npz,
        layer_names=np.array(layer_names, dtype=object),
        bits_ladder=np.array(ladder), reference_bits=np.array([ref_bits]),
        delta_loss_a=out["A"]["loss"].delta, delta_loss_b=out["B"]["loss"].delta,
        delta_acc_a=out["A"]["acc"].delta, delta_acc_b=out["B"]["acc"].delta,
        baseline_loss_a=np.array([out["A"]["loss"].baseline]),
        baseline_loss_b=np.array([out["B"]["loss"].baseline]),
        baseline_acc_a=np.array([out["A"]["acc"].baseline]),
        baseline_acc_b=np.array([out["B"]["acc"].baseline]),
    )
    dump_json(outdir / f"rank_{tag}.json", {
        "backbone": name, "target_domain": shift, "reference_bits": ref_bits,
        "n_layers": len(layer_names), "bits_ladder": ladder,
        "layer_names": layer_names,
        "sensitivity_loss": {"A": sl_a, "B": sl_b},
        "sensitivity_acc": {"A": sa_a, "B": sa_b},
        "baseline": {
            "loss_A": out["A"]["loss"].baseline, "loss_B": out["B"]["loss"].baseline,
            "acc_A": out["A"]["acc"].baseline, "acc_B": out["B"]["acc"].baseline,
        },
        "rank_correlation": {"loss": res_loss.as_row(), "acc": res_acc.as_row()},
        "elapsed_s": round(time.time() - t0, 2),
    })
    print(f"\n  wrote {npz}")
    print(f"  wrote {outdir / f'rank_{tag}.json'}")
    print(f"  elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
