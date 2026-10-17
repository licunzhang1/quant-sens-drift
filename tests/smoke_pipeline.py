"""End-to-end smoke test: run the whole experiment pipeline with a pure numpy reference model.

The chain it exercises (structurally identical to the real torch experiments):

    synthetic IQ (Domain A / Domain B)
      -> classic AMC cumulant features
      -> train the reference model
      -> measure both domains (look at the domain gap)
      -> per-layer selective quantization -> sensitivity matrix (loss + acc metric)
      -> Spearman rho (with a permutation test)
      -> equal reconstruction error three arms
      -> write results to disk

How to run:

    python tests/smoke_pipeline.py
    python tests/smoke_pipeline.py --raw-iq    # reproduce "raw IQ into an MLP fails"

======================================================================
Two measured findings this script records (the real experiments must heed them)
======================================================================

[Finding 1] Feeding raw IQ straight into an MLP does not work -- it is a
representation problem, not a data problem
------------------------------------------------------------------
Training accuracy 1.0, test 0.50; per class only BPSK is separable (0.98), while
QPSK/8PSK/16QAM all sit at 0.33.
The reason: the modulation information lives in the **distribution** of the
samples rather than at a few particular sampling instants, and a plain MLP cannot
learn it. After switching to classic cumulant features, both the in-domain
holdout and a completely fresh dataset give 1.0000.
-> see tests/diagnose_data.py and tests/diagnose_features.py

[Finding 2] Accuracy saturates, so **loss must be the primary sensitivity metric**
------------------------------------------------------------------
When the in-domain accuracy is already 1.0000, quantizing one layer to 2 bit can
still leave it at 1.0000. Then Delta acc == 0, the sensitivity vector is all
zeros, and the rank correlation degenerates straight to NaN.
Accuracy is a **discrete quantity with a saturation ceiling**, and it carries too
little information for a quantization sensitivity study.
-> so this script computes both loss and acc sensitivities, **uses the loss
   metric for rho**, and prints the two side by side so the difference is visible.

In the real experiment config (configs/base.yaml) ``sensitivity.metric`` already
defaults to ``loss``.

[Finding 3] The representation decides **which domain shift is fatal**
------------------------------------------------------------------
Fixing the same SNR shift (6,14)->(0,8) dB and varying only the carrier frequency
offset CFO:

    CFO=0.000 -> target acc 0.6875
    CFO=0.001 -> target acc 0.4288
    CFO=0.005 -> target acc 0.3300
    CFO=0.020 -> target acc 0.3137   <- drops to chance level

The reason is that higher-order cumulants are **not CFO invariant**: under CFO
``E[z^4]`` picks up an ``exp(j8*pi*f*k)`` phase, and with L=512, f=0.02 the phase
wraps 82 times and the cumulant cancels to zero.

This is not a bug but a phenomenon worth putting in the report: **the "strength"
of a domain shift is not an absolute quantity; it depends on the input
representation**. The same CFO is catastrophic for cumulant features, while for a
CNN able to learn phase invariance it may only cause mild degradation. The
factorized protocol of this project captures exactly this.
-> see tests/diagnose_regime.py

That is why the smoke test uses only the **SNR shift** (CFO=0) by default, so the
chain can be exercised in a usable range where the source domain is not saturated
and the target domain does not collapse.
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.equal_error import run_equal_error_arm  # noqa: E402
from src.analysis.proxies import correlate_with_sensitivity, proxy_matrix  # noqa: E402
from src.analysis.rank_corr import analyse_pair  # noqa: E402
from src.analysis.sensitivity import compute_sensitivity_matrix  # noqa: E402
from src.data.features import FEATURE_NAMES, amc_features  # noqa: E402
from src.data.synth import ChannelConfig, make_iq_dataset  # noqa: E402
from src.models.numpy_mlp import NumpyMLP, quantized_layer  # noqa: E402
from src.quant.core import quant_error_curve  # noqa: E402
from src.utils import dump_json  # noqa: E402

BITS_LADDER = [2, 3, 4, 6, 8]
REFERENCE_BITS = 4
MODS = ("bpsk", "qpsk", "8psk", "16qam")


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def print_matrix(title: str, layer_names, bits, mat, fmt: str = "{:>12.5f}") -> None:
    print(f"\n  {title}")
    print("    {:<12s}".format("layer") + "".join(f"{b:>9d}bit" for b in bits))
    for i, name in enumerate(layer_names):
        row = "    {:<12s}".format(name)
        for j in range(len(bits)):
            row += fmt.format(mat[i, j])
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="quant-sens end-to-end smoke test (pure numpy)")
    ap.add_argument("--raw-iq", action="store_true",
                    help="feed raw IQ straight into the MLP (expected to fail; "
                         "reproduces the representation problem)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="results/smoke")
    args = ap.parse_args()

    n_train, n_test, n_sym, sps, epochs = 3000, 1000, 64, 4, 100
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------------ data
    banner("step 1/6  generate synthetic IQ (Domain A and Domain B)")
    # The difficulty comes from the usable range that tests/diagnose_regime.py
    # scans out: the source domain is not saturated (acc~=0.95) and the target
    # domain does not collapse (acc~=0.56).
    dom_a = ChannelConfig.from_dict({"snr_db": (4.0, 12.0)})
    # Only the SNR shift. CFO is catastrophic for cumulant features (see
    # [finding 3] at the top of the file); it drops the target domain straight to
    # chance level and the sensitivity ranking then turns into noise.
    dom_b = ChannelConfig.from_dict({"snr_db": (-2.0, 6.0)})

    t0 = time.time()
    Xa, ya = make_iq_dataset(n_train, dom_a, modulations=MODS, n_symbols=n_sym, sps=sps,
                             seed=args.seed)
    Xa_te, ya_te = make_iq_dataset(n_test, dom_a, modulations=MODS, n_symbols=n_sym, sps=sps,
                                   seed=args.seed + 1000)
    Xb_te, yb_te = make_iq_dataset(n_test, dom_b, modulations=MODS, n_symbols=n_sym, sps=sps,
                                   seed=args.seed + 2000)
    print(f"  Domain A train : {Xa.shape}   {dom_a.describe()}")
    print(f"  Domain A test  : {Xa_te.shape}")
    print(f"  Domain B test  : {Xb_te.shape}   {dom_b.describe()}")
    print(f"  elapsed {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ representation
    banner("step 2/6  input representation")
    if args.raw_iq:
        Fa, Fa_te, Fb_te = (x.reshape(x.shape[0], -1) for x in (Xa, Xa_te, Xb_te))
        repr_name = "raw_iq"
        print(f"  raw IQ flattened: {Fa.shape[1]} dims")
        print("  Expected: very high training accuracy but in-domain test near "
              "chance -- the representation carries no distributional information.")
    else:
        Fa, Fa_te, Fb_te = amc_features(Xa), amc_features(Xa_te), amc_features(Xb_te)
        repr_name = "amc_cumulants"
        print(f"  classic AMC cumulant features: {Fa.shape[1]} dims")
        print(f"  {', '.join(FEATURE_NAMES)}")

    mu, sd = Fa.mean(axis=0), Fa.std(axis=0) + 1e-9
    Fa, Fa_te, Fb_te = [(F - mu) / sd for F in (Fa, Fa_te, Fb_te)]
    print("  standardized with Domain A training statistics (no test-set information)")

    # ------------------------------------------------------------------ training
    banner("step 3/6  train the reference model on Domain A")
    in_dim = Fa.shape[1]
    if args.raw_iq:
        hidden = (128, 64)
    else:
        # deliberately a bit deep: with too few layers the rank correlation has
        # almost no statistical power
        hidden = (48, 48, 40, 40, 32, 32, 24)
    dims = [in_dim, *hidden, len(MODS)]
    model = NumpyMLP(dims, seed=args.seed)
    n_params = sum(v.size for v in model.params.values())
    print(f"  dims     = {dims}")
    print(f"  params   = {n_params:,}    weight layers = {len(model.weight_layer_names())}")
    print("  training ...")
    t0 = time.time()
    hist = model.train(Fa, ya, epochs=epochs, batch_size=128, lr=3e-3,
                       weight_decay=1e-4, seed=args.seed, verbose=False)
    print(f"  elapsed {time.time() - t0:.1f}s   loss {hist[0]:.4f} -> {hist[-1]:.4f}")

    # ------------------------------------------------------------------ domain gap
    banner("step 4/6  metrics on both domains (domain gap)")
    acc_a_tr = model.accuracy(Fa, ya)
    acc_a = model.accuracy(Fa_te, ya_te)
    acc_b = model.accuracy(Fb_te, yb_te)
    loss_a = model.loss(Fa_te, ya_te)
    loss_b = model.loss(Fb_te, yb_te)
    print(f"  Domain A train set     : acc={acc_a_tr:.4f}")
    print(f"  Domain A test  (in-dom): acc={acc_a:.4f}   loss={loss_a:.4f}")
    print(f"  Domain B test  (target): acc={acc_b:.4f}   loss={loss_b:.4f}")
    print(f"  in-domain generalization gap = {acc_a_tr - acc_a:+.4f}   (should be near 0)")
    print(f"  cross-domain drop            = {acc_a - acc_b:+.4f}   "
          f"loss rise = {loss_b - loss_a:+.4f}")
    if acc_a_tr - acc_a > 0.05:
        print("  [!] the in-domain gap is large: the domain gap will be confounded "
              "with a generalization failure.")
    if acc_b < 0.30:
        print("  [!] the target domain is near chance level: the sensitivity ranking "
              "is itself noise, so the shift should be weakened.")
    if acc_a > 0.999:
        print("  [!] source-domain accuracy is saturated at 1.0: the Delta acc metric "
              "degenerates to all zeros (see the comparison below).")

    # ------------------------------------------------------------------ sensitivity
    banner("step 5/6  per-layer sensitivity matrix (loss metric + acc metric)")
    layer_names = model.weight_layer_names()
    ev_a = lambda: model.loss(Fa_te, ya_te)      # noqa: E731
    ev_b = lambda: model.loss(Fb_te, yb_te)      # noqa: E731
    ac_a = lambda: model.accuracy(Fa_te, ya_te)  # noqa: E731
    ac_b = lambda: model.accuracy(Fb_te, yb_te)  # noqa: E731
    ql = lambda name, b: quantized_layer(model, name, b, per_channel=True)  # noqa: E731

    print("\n  [Domain A] loss metric")
    sens_a = compute_sensitivity_matrix(layer_names, BITS_LADDER, evaluate=ev_a,
                                        quantize_layer=ql, metric="loss",
                                        higher_is_better=False, domain="A",
                                        backbone="numpy_mlp")
    print("\n  [Domain B] loss metric")
    sens_b = compute_sensitivity_matrix(layer_names, BITS_LADDER, evaluate=ev_b,
                                        quantize_layer=ql, metric="loss",
                                        higher_is_better=False, domain="B",
                                        backbone="numpy_mlp")
    print("\n  [Domain A] acc metric (shown to demonstrate the saturation problem)")
    accsens_a = compute_sensitivity_matrix(layer_names, BITS_LADDER, evaluate=ac_a,
                                           quantize_layer=ql, metric="acc",
                                           higher_is_better=True, domain="A",
                                           backbone="numpy_mlp")
    print("\n  [Domain B] acc metric")
    accsens_b = compute_sensitivity_matrix(layer_names, BITS_LADDER, evaluate=ac_b,
                                           quantize_layer=ql, metric="acc",
                                           higher_is_better=True, domain="B",
                                           backbone="numpy_mlp")

    print_matrix("Delta loss (larger = more sensitive; primary metric)",
                 layer_names, BITS_LADDER, sens_a.delta)
    print_matrix("Delta loss (Domain B)", layer_names, BITS_LADDER, sens_b.delta)
    print_matrix("Delta acc (Domain A; note the saturation)", layer_names, BITS_LADDER,
                 accsens_a.delta, "{:>12.4f}")
    print_matrix("Delta acc (Domain B)", layer_names, BITS_LADDER, accsens_b.delta,
                 "{:>12.4f}")

    # ------------------------------------------------------------------ rho
    banner("step 6/6  rank stability rho (the core quantity)")
    sa = sens_a.sensitivity_at(REFERENCE_BITS)
    sb = sens_b.sensitivity_at(REFERENCE_BITS)
    ta = accsens_a.sensitivity_at(REFERENCE_BITS)
    tb = accsens_b.sensitivity_at(REFERENCE_BITS)
    print(f"  reference bits = {REFERENCE_BITS}   layers n = {len(layer_names)}")
    print(f"  Domain A sensitivity (loss): {np.round(sa, 5).tolist()}")
    print(f"  Domain B sensitivity (loss): {np.round(sb, 5).tolist()}")

    res = analyse_pair(sa, sb, n_permutations=20000, seed=args.seed)
    print(f"\n  === loss metric (primary) ===")
    print(f"  rho (Spearman) = {res.rho:+.4f}    p = {res.p_value:.4g}")
    print(f"  95% CI = [{res.ci_low:+.3f}, {res.ci_high:+.3f}]   "
          f"Kendall tau-b = {res.tau_b:+.4f}")
    print(f"  verdict: {res.verdict()}")

    res_acc = analyse_pair(ta, tb, n_permutations=20000, seed=args.seed)
    print(f"\n  === acc metric (control, showing the consequence of saturation) ===")
    print(f"  rho (Spearman) = {res_acc.rho:+.4f}    p = {res_acc.p_value:.4g}")
    print(f"  verdict: {res_acc.verdict()}")
    print(f"  distinct values in the A-domain acc sensitivity = "
          f"{int(np.unique(ta).size)} / {ta.size}")

    # ------------------------------------------------------------------ equal error arm
    banner("appendix 1: equal reconstruction error arms (three-arm comparison)")
    curves = {n: quant_error_curve(model.params[n], BITS_LADDER, True, 0)
              for n in layer_names}
    err = np.array([[curves[n][b]["nmse"] for b in BITS_LADDER] for n in layer_names])
    print_matrix("per-layer NMSE eps_l(b)", layer_names, BITS_LADDER, err)

    costs = model.layer_costs()
    arm = run_equal_error_arm(sa, sb, err, BITS_LADDER, budget_ratio=0.6,
                              layer_costs=costs, layer_names=layer_names,
                              source_domain="A", target_domain="B")
    print(f"\n  budget ratio = {arm.budget_ratio}   total budget = {arm.budget_total:.1f}")
    print("    {:<20s}{:<26s}{:>16s}".format("arm", "bits", "pred_damage"))
    for tag, alloc in (("uniform", arm.uniform),
                       ("agnostic (source A)", arm.agnostic),
                       ("oracle (target B)", arm.oracle)):
        print("    {:<20s}{:<26s}{:>16.8f}".format(
            tag, str(alloc.bits.tolist()), alloc.predicted_damage))
    print(f"\n  gain from sensitivity guidance (uniform - agnostic) = "
          f"{arm.gain_from_sensitivity:+.8f}")
    print(f"  **cost of not knowing the deployment domain** (agnostic - oracle) = "
          f"{arm.cost_of_domain_ignorance:+.8f}")
    print(f"  relative cost = {arm.relative_cost_of_ignorance:+.1%}")
    print(f"  ranking flips the decision: {arm.ranking_flipped}")
    print(f"  verdict: {arm.verdict()}")

    # ------------------------------------------------------------------ proxies
    banner("appendix 2: can proxies predict the sensitivity (bonus)")
    tensors = {n: model.params[n] for n in layer_names}
    prox, _ = proxy_matrix(tensors, bits=REFERENCE_BITS)
    rows = correlate_with_sensitivity(prox, sb, n_permutations=20000, seed=args.seed)
    for r in rows[:6]:
        flag = " *" if r["significant"] else ""
        print(f"    {r['proxy']:<14s} rho={r['rho']:+.4f}  p={r['p_value']:.4f}{flag}")
    print("  (* marks p<0.05. With layer counts in the single digits, treat the "
          "significance of any single proxy with caution.)")

    # ------------------------------------------------------------------ write results
    banner("writing results")
    tag = ("full" if not args.raw_iq else "rawiq")
    summary = {
        "config": {
            "seed": args.seed, "representation": repr_name, "dims": dims,
            "n_params": int(n_params), "epochs": epochs, "bits_ladder": BITS_LADDER,
            "reference_bits": REFERENCE_BITS,
            "domain_a": dom_a.describe(), "domain_b": dom_b.describe(),
            "n_train": n_train, "n_test": n_test, "n_symbols": n_sym, "sps": sps,
            "primary_metric": "loss",
        },
        "domain_gap": {
            "acc_a_train": acc_a_tr, "acc_a_test": acc_a, "acc_b_test": acc_b,
            "indomain_gap": acc_a_tr - acc_a, "cross_domain_drop": acc_a - acc_b,
            "loss_a": loss_a, "loss_b": loss_b,
        },
        "layer_names": layer_names,
        "sensitivity_loss": {"A": sens_a.delta, "B": sens_b.delta,
                             "baseline_A": sens_a.baseline, "baseline_B": sens_b.baseline},
        "sensitivity_acc": {"A": accsens_a.delta, "B": accsens_b.delta,
                            "baseline_A": accsens_a.baseline, "baseline_B": accsens_b.baseline},
        "rank_correlation": {"loss": res.as_row(), "acc": res_acc.as_row()},
        "err_curves": err,
        "equal_error_arm": {
            "budget_ratio": arm.budget_ratio, "budget_total": arm.budget_total,
            "uniform_bits": arm.uniform.bits, "agnostic_bits": arm.agnostic.bits,
            "oracle_bits": arm.oracle.bits,
            "uniform_damage": arm.uniform.predicted_damage,
            "agnostic_damage": arm.agnostic.predicted_damage,
            "oracle_damage": arm.oracle.predicted_damage,
            "gain_from_sensitivity": arm.gain_from_sensitivity,
            "cost_of_domain_ignorance": arm.cost_of_domain_ignorance,
            "relative_cost_of_ignorance": arm.relative_cost_of_ignorance,
            "ranking_flipped": arm.ranking_flipped,
            "verdict": arm.verdict(),
        },
        "proxies_vs_sensitivity": rows,
        "elapsed_s": round(time.time() - t_start, 2),
    }
    p_json = dump_json(outdir / f"smoke_{tag}.json", summary)
    np.savez_compressed(
        outdir / f"sensitivity_{tag}.npz",
        delta_loss_a=sens_a.delta, delta_loss_b=sens_b.delta,
        delta_acc_a=accsens_a.delta, delta_acc_b=accsens_b.delta,
        err_curves=err, costs=costs,
        layer_names=np.array(layer_names, dtype=object),
        bits_ladder=np.array(BITS_LADDER), scaler_mu=mu, scaler_sd=sd)
    print(f"  wrote {p_json}")
    print(f"  wrote {outdir / f'sensitivity_{tag}.npz'}")

    banner("summary")
    print(f"  pipeline ran end to end in {time.time() - t_start:.1f}s")
    print(f"  representation          : {repr_name}")
    print(f"  in-domain gap           : {acc_a_tr - acc_a:+.4f}")
    print(f"  cross-domain drop (dAcc): {acc_a - acc_b:+.4f}")
    print(f"  rank rho (loss metric)  : {res.rho:+.4f}  (p={res.p_value:.4g})")
    print(f"  rank rho (acc metric)   : {res_acc.rho:+.4f}  (p={res_acc.p_value:.4g})")
    print(f"  cost of domain ignorance: {arm.cost_of_domain_ignorance:+.8f}")
    print()
    print("  This script only checks that the chain works; it produces no "
          "scientific conclusion:")
    print("  - the reference model is not a real backbone "
          "(real experiments use CLDNN / 1D-ResNet)")
    print("  - only two shift types and a single backbone are used, so the "
          "estimate of rho has a large variance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
