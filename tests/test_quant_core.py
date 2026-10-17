"""Unit tests for the core quantization math and rank-correlation analysis.

Depends on numpy only; **torch is not needed**. Run it directly:

    python tests/test_quant_core.py

or through pytest.
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

import pathlib
import sys
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quant.core import (  # noqa: E402
    channel_wise_nmse,
    quant_error_curve,
    quant_params,
    quantize,
    quantize_dequantize,
    qrange,
    reconstruction_error,
)
from src.analysis.rank_corr import (  # noqa: E402
    bootstrap_ci,
    kendall_tau_b,
    permutation_test,
    rankdata_average,
    spearman,
)
from src.analysis.equal_error import (  # noqa: E402
    greedy_allocate,
    predicted_damage,
    run_equal_error_arm,
    uniform_allocate,
)

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    line = f"  [{mark}] {name}"
    if detail and not cond:
        line += f"  <- {detail}"
    print(line)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ==========================================================================
# 1. quantization range
# ==========================================================================
def test_qrange() -> None:
    print("\n[1] quantization range")
    check("qrange(8, sym) == (-127, 127)", qrange(8, True) == (-127, 127), str(qrange(8, True)))
    check("qrange(8, asym) == (0, 255)", qrange(8, False) == (0, 255), str(qrange(8, False)))
    check("qrange(2, sym) == (-1, 1)", qrange(2, True) == (-1, 1), str(qrange(2, True)))
    check("qrange(4, asym) == (0, 15)", qrange(4, False) == (0, 15), str(qrange(4, False)))

    raised = False
    try:
        qrange(1, True)
    except ValueError:
        raised = True
    check("qrange(1, sym) raises ValueError (qmax would be 0)", raised)

    raised = False
    try:
        qrange(0, False)
    except ValueError:
        raised = True
    check("qrange(0, asym) raises ValueError", raised)


# ==========================================================================
# 2. quantization correctness
# ==========================================================================
def test_exact_quantization() -> None:
    print("\n[2] exactly representable tensors must give zero error")
    x = np.array([-1.0, 0.0, 1.0])
    xq, scale, zero = quantize_dequantize(x, 2, symmetric=True)
    check("x=[-1,0,1], 2bit symmetric -> no error", approx(np.max(np.abs(x - xq)), 0.0),
          f"max_err={np.max(np.abs(x - xq)):.3e}, scale={scale}, zero={zero}")

    err = reconstruction_error(x, xq)
    check("reconstruction error nmse == 0", approx(err["nmse"], 0.0, 1e-12), str(err))

    # integer quantization codes must fall inside the range
    rng = np.random.default_rng(0)
    y = rng.standard_normal(1000) * 5
    for bits in (2, 3, 4, 6, 8):
        qmin, qmax = qrange(bits, True)
        q = quantize(y, bits, symmetric=True)
        ok = (q.min() >= qmin - 1e-9) and (q.max() <= qmax + 1e-9)
        check(f"  quantization codes inside the range (bits={bits})", ok,
              f"[{q.min()},{q.max()}] vs [{qmin},{qmax}]")


def test_error_monotone_in_bits() -> None:
    print("\n[3] reconstruction error decreases monotonically with bit width")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 32)) * 2.0
    ladder = [2, 3, 4, 6, 8]
    errs = [reconstruction_error(x, quantize_dequantize(x, b, True)[0])["nmse"] for b in ladder]
    print("      nmse:", "  ".join(f"{b}bit={e:.4f}" for b, e in zip(ladder, errs)))
    mono = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    check("nmse strictly decreasing", mono, str(errs))
    check("nmse at 8bit is far below 2bit", errs[-1] < errs[0] / 10,
          f"{errs[-1]:.5f} vs {errs[0]:.5f}")


def test_per_channel_vs_per_tensor() -> None:
    print("\n[4] per-channel is no worse than per-tensor "
          "(clearly better when channel scales differ a lot)")
    rng = np.random.default_rng(2)
    # build a weight matrix whose channel scales differ wildly
    # (very common in real networks)
    scales = np.array([0.01, 0.1, 1.0, 10.0])[:, None]
    W = rng.standard_normal((4, 64)) * scales

    for bits in (2, 4, 8):
        e_pt = reconstruction_error(W, quantize_dequantize(W, bits, True, None)[0])["nmse"]
        e_pc = reconstruction_error(W, quantize_dequantize(W, bits, True, 0)[0])["nmse"]
        check(
            f"  bits={bits}: per-channel <= per-tensor",
            e_pc <= e_pt + 1e-12,
            f"pc={e_pc:.5f} pt={e_pt:.5f}",
        )

    # The global NMSE is energy weighted, so large-scale channels dominate the
    # number and hide the fact that small-scale channels get destroyed by
    # quantization. Only a per-channel unweighted NMSE makes it visible.
    xq_pt = quantize_dequantize(W, 4, True, None)[0]
    xq_pc = quantize_dequantize(W, 4, True, 0)[0]
    cw_pt = channel_wise_nmse(W, xq_pt, axis=0)
    cw_pc = channel_wise_nmse(W, xq_pc, axis=0)
    print(f"      per-channel unweighted NMSE @4bit: per-tensor={cw_pt:.4f}  "
          f"per-channel={cw_pc:.4f}")
    check("  per-channel unweighted NMSE: per-channel is at least 10x better",
          cw_pc * 10 < cw_pt,
          f"pc={cw_pc:.6f} pt={cw_pt:.6f} ratio={cw_pt / max(cw_pc, 1e-15):.1f}x")

    # and under per-channel every channel should be represented reasonably
    # (no channel should be destroyed outright)
    ax = 1
    per_ch = np.mean((W - xq_pc) ** 2, axis=ax) / (np.mean(W ** 2, axis=ax) + 1e-12)
    check("  no channel is completely destroyed at 4bit per-channel",
          bool(np.all(per_ch < 0.5)), str(np.round(per_ch, 4).tolist()))


def test_asymmetric_for_skewed() -> None:
    print("\n[5] asymmetric beats symmetric on one-sided distributions")
    rng = np.random.default_rng(3)
    x = rng.exponential(scale=1.0, size=20000)  # all positive, long tailed
    for bits in (3, 4, 6):
        e_sym = reconstruction_error(x, quantize_dequantize(x, bits, True)[0])["nmse"]
        e_asym = reconstruction_error(x, quantize_dequantize(x, bits, False)[0])["nmse"]
        check(
            f"  bits={bits}: asymmetric <= symmetric",
            e_asym <= e_sym + 1e-12,
            f"sym={e_sym:.6f} asym={e_asym:.6f}",
        )


def test_edge_cases() -> None:
    print("\n[6] edge cases")
    z = np.zeros((8, 8))
    xq, scale, zero = quantize_dequantize(z, 4, True)
    check("all-zero tensor produces no NaN/Inf", np.all(np.isfinite(xq)), str(xq.flat[:3]))

    const = np.full((4, 4), 3.5)
    xq2, _, _ = quantize_dequantize(const, 4, True)
    check("constant tensor produces no NaN/Inf", np.all(np.isfinite(xq2)))

    g = np.linspace(-1, 1, 1000).reshape(10, 100)
    scale_s, zero_s = quant_params(g, 4, True, None)
    scale_c, zero_c = quant_params(g, 4, True, 0)
    check("per-channel scale shape is broadcastable", scale_c.shape == (10, 1), str(scale_c.shape))
    check("per-tensor scale is scalar-like", np.size(scale_s) == 1, str(np.shape(scale_s)))
    check("per-channel zero_point shape is consistent", zero_c.shape == (10, 1), str(zero_c.shape))
    check("symmetric quantization always has zero_point 0", float(np.max(np.abs(zero_s))) == 0.0
          and float(np.max(np.abs(zero_c))) == 0.0)

    raised = False
    try:
        quant_params(np.array([]), 4)
    except ValueError:
        raised = True
    check("empty array raises ValueError", raised)


def test_error_curve() -> None:
    print("\n[7] error curve interface")
    rng = np.random.default_rng(4)
    x = rng.standard_normal((16, 32))
    curve = quant_error_curve(x, [2, 4, 8])
    check("returns 3 bit widths", len(curve) == 3, str(list(curve.keys())))
    check("every entry has nmse", all("nmse" in v for v in curve.values()))
    check("nmse decreases with bit width", curve[8]["nmse"] < curve[4]["nmse"] < curve[2]["nmse"],
          str({k: round(v["nmse"], 5) for k, v in curve.items()}))

    # bit width < 2 with symmetric=True -> skip rather than crash
    curve2 = quant_error_curve(x, [1, 2, 4], symmetric=True)
    check("bits=1 is skipped in symmetric mode", 1 not in curve2, str(list(curve2.keys())))


# ==========================================================================
# 8. rank correlation
# ==========================================================================
def test_rankdata() -> None:
    print("\n[8] average ranks")
    r = rankdata_average([3.0, 1.0, 2.0])
    check("no ties: [3,1,2] -> [3,1,2]", np.allclose(r, [3, 1, 2]), str(r))

    r = rankdata_average([10.0, 20.0, 20.0, 30.0])
    check("with ties: average ranks -> [1, 2.5, 2.5, 4]", np.allclose(r, [1, 2.5, 2.5, 4]), str(r))

    r = rankdata_average([5.0, 5.0, 5.0])
    check("all tied -> all 2", np.allclose(r, [2, 2, 2]), str(r))

    rng = np.random.default_rng(5)
    a = rng.standard_normal(50)
    check("ranks agree with the sort order",
          np.array_equal(np.argsort(rankdata_average(a)), np.argsort(a)))


def test_spearman() -> None:
    print("\n[9] Spearman")
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    check("strictly increasing -> rho=1", approx(spearman(a, a ** 3), 1.0, 1e-9),
          str(spearman(a, a ** 3)))
    check("strictly decreasing -> rho=-1", approx(spearman(a, -a), -1.0, 1e-9),
          str(spearman(a, -a)))

    b = a + 100.0
    check("shift invariant -> rho=1", approx(spearman(a, b), 1.0, 1e-9), str(spearman(a, b)))

    c = np.exp(a)
    check("nonlinear but monotone -> rho=1", approx(spearman(a, c), 1.0, 1e-9),
          str(spearman(a, c)))

    # cross-check against scipy (when installed)
    try:
        from scipy.stats import spearmanr  # type: ignore

        rng = np.random.default_rng(6)
        for _ in range(5):
            u = rng.standard_normal(20)
            v = rng.standard_normal(20)
            ours = spearman(u, v)
            theirs = float(spearmanr(u, v).statistic)
            if not approx(ours, theirs, 1e-9):
                check("agrees with scipy.stats.spearmanr", False, f"{ours} vs {theirs}")
                return
        check("agrees with scipy.stats.spearmanr (5 random samples)", True)
    except ImportError:
        check("cross-check against scipy (scipy not installed, skipped)", True, "skipped")


def test_permutation_test() -> None:
    print("\n[10] permutation test")
    rng = np.random.default_rng(7)

    x = rng.standard_normal(12)
    # strongly correlated
    rho, p = permutation_test(x, x * 2 + 1, n_permutations=2000, seed=1)
    check("perfectly correlated -> rho~=1, p tiny",
          approx(rho, 1.0, 1e-9) and p < 0.01, f"rho={rho}, p={p}")

    # fully independent: repeat and check the p-values are roughly uniform
    # (they should not always be small)
    ps = []
    for s in range(20):
        y = rng.standard_normal(12)
        _, pv = permutation_test(x, y, n_permutations=500, seed=100 + s)
        ps.append(pv)
    frac_small = float(np.mean(np.array(ps) < 0.05))
    check(
        "independent data -> rejection rate near the nominal 5% "
        "(significance is not badly overstated)",
        frac_small <= 0.30,
        f"{int(frac_small * 20)} of 20 runs have p<0.05, "
        f"p-values={np.round(ps, 3).tolist()}",
    )

    lo, hi = bootstrap_ci(x, x * 2 + 1, n_boot=500, seed=2)
    check("CI converges to 1 for perfect correlation",
          lo > 0.9 and hi <= 1.0 + 1e-9, f"[{lo:.3f}, {hi:.3f}]")

    check("Kendall tau-b perfectly concordant -> 1",
          approx(kendall_tau_b(x, x * 2 + 1), 1.0, 1e-9))
    check("Kendall tau-b perfectly reversed -> -1",
          approx(kendall_tau_b(x, -x), -1.0, 1e-9))


# ==========================================================================
# 11. equal reconstruction error arms
# ==========================================================================
def _fake_err_curves(n_layers: int, ladder=(2, 3, 4, 6, 8), seed=0):
    """Build a plausible error curve: decreasing in bit width, varied across layers."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 0.5, size=(n_layers, 1))
    shape = np.array([[1.0, 0.6, 0.35, 0.15, 0.05]], dtype=np.float64)
    return base * shape


def test_greedy_budget() -> None:
    print("\n[11] greedy allocation respects the budget")
    E = _fake_err_curves(8, seed=11)
    s = np.linspace(0.1, 1.0, 8)
    costs = np.ones(8)

    for ratio in (0.3, 0.5, 0.8, 1.0):
        alloc = greedy_allocate(s, E, [2, 3, 4, 6, 8], budget_ratio=ratio, layer_costs=costs)
        check(
            f"  ratio={ratio}: budget not exceeded",
            alloc.budget_used <= alloc.budget_total + 1e-6,
            f"used={alloc.budget_used:.2f} total={alloc.budget_total:.2f}",
        )
        check(
            f"  ratio={ratio}: every bit width is on the ladder",
            set(np.unique(alloc.bits)).issubset({2, 3, 4, 6, 8}),
            str(alloc.bits.tolist()),
        )

    # a larger budget gives a higher mean bit width
    means = [
        greedy_allocate(s, E, [2, 3, 4, 6, 8], budget_ratio=r, layer_costs=costs).bits.mean()
        for r in (0.2, 0.5, 0.9)
    ]
    check("larger budget gives a higher mean bit width",
          means[0] <= means[1] <= means[2], str(means))


def test_greedy_puts_bits_where_sensitive() -> None:
    print("\n[12] greedy gives bits to the sensitive layers first")
    E = _fake_err_curves(6, seed=12)
    costs = np.ones(6)
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0])  # only the last layer is sensitive

    alloc = greedy_allocate(s, E, [2, 3, 4, 6, 8], budget_ratio=0.5, layer_costs=costs)
    check("the sensitive layer gets the highest bit width",
          alloc.bits[-1] == alloc.bits.max(), str(alloc.bits.tolist()))
    check("insensitive layers stay at the lowest bit width",
          np.all(alloc.bits[:-1] == 2), str(alloc.bits.tolist()))

    # when all sensitivities are equal the allocation should be near uniform
    s_eq = np.ones(6)
    alloc_eq = greedy_allocate(s_eq, E, [2, 3, 4, 6, 8], budget_ratio=0.6, layer_costs=costs)
    spread = int(alloc_eq.bits.max() - alloc_eq.bits.min())
    check("equal sensitivities give a nearly uniform allocation "
          "(max-min bit width <= 2 rungs)", spread <= 4,
          str(alloc_eq.bits.tolist()))


def test_noise_floor() -> None:
    print("\n[13] negative sensitivity is clipped (so bits are not moved "
          "away from noisy layers)")
    E = _fake_err_curves(5, seed=13)
    # all pure noise where quantizing happens to help
    s = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
    alloc = greedy_allocate(s, E, [2, 3, 4, 6, 8], budget_ratio=0.9, layer_costs=np.ones(5))
    check("all-negative sensitivity -> everything stays at the lowest bit "
          "width (no wasted budget)",
          np.all(alloc.bits == 2), str(alloc.bits.tolist()))

    s_mix = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    alloc_mix = greedy_allocate(s_mix, E, [2, 3, 4, 6, 8], budget_ratio=0.6, layer_costs=np.ones(5))
    check("layers with negative/zero sensitivity get no extra bit width",
          np.all(alloc_mix.bits[:2] == 2), str(alloc_mix.bits.tolist()))


def test_uniform_allocation() -> None:
    print("\n[14] uniform allocation")
    E = _fake_err_curves(4, seed=14)
    s = np.ones(4)
    alloc = uniform_allocate(4, [2, 3, 4, 6, 8], budget_ratio=0.5,
                             layer_costs=np.ones(4), sensitivity=s, err_curves=E)
    check("all layers get the same bit width",
          len(set(alloc.bits.tolist())) == 1, str(alloc.bits.tolist()))
    check("budget not exceeded", alloc.budget_used <= alloc.budget_total + 1e-6,
          f"{alloc.budget_used} <= {alloc.budget_total}")


def test_equal_error_arm() -> None:
    print("\n[15] three-arm comparison")
    E = _fake_err_curves(10, seed=15)
    ladder = [2, 3, 4, 6, 8]
    costs = np.ones(10)

    # (a) identical sensitivities in both domains -> agnostic == oracle;
    #     not knowing the domain costs nothing
    s = np.linspace(0.1, 1.0, 10)
    r_same = run_equal_error_arm(s, s.copy(), E, ladder, 0.6, costs)
    check("agnostic == oracle when the sensitivities match",
          np.array_equal(r_same.agnostic.bits, r_same.oracle.bits),
          f"{r_same.agnostic.bits.tolist()} vs {r_same.oracle.bits.tolist()}")
    check("cost of domain ignorance ~= 0 when the sensitivities match",
          abs(r_same.cost_of_domain_ignorance) < 1e-12,
          str(r_same.cost_of_domain_ignorance))

    # (b) fully reversed sensitivities -> should give a positive cost
    s_rev = s[::-1].copy()
    r_rev = run_equal_error_arm(s, s_rev, E, ladder, 0.6, costs)
    check("agnostic != oracle when the ranking is reversed",
          not np.array_equal(r_rev.agnostic.bits, r_rev.oracle.bits),
          f"{r_rev.agnostic.bits.tolist()} vs {r_rev.oracle.bits.tolist()}")
    check("cost of domain ignorance > 0 when the ranking is reversed",
          r_rev.cost_of_domain_ignorance > 0,
          str(r_rev.cost_of_domain_ignorance))
    check("ranking_flipped flag is set correctly", r_rev.ranking_flipped is True)
    print(f"      verdict: {r_rev.verdict()}")

    # (c) consistency of the predicted damage
    d = predicted_damage(r_rev.oracle.bits, s_rev, E, ladder)
    check("the arm damage agrees when computed with the target-domain sensitivity",
          approx(d, r_rev.oracle.predicted_damage, 1e-9),
          f"{d} vs {r_rev.oracle.predicted_damage}")


# ==========================================================================
def main() -> int:
    print("=" * 74)
    print("  quant-sens core unit tests (pure numpy, no torch needed)")
    print("=" * 74)

    tests = [
        test_qrange,
        test_exact_quantization,
        test_error_monotone_in_bits,
        test_per_channel_vs_per_tensor,
        test_asymmetric_for_skewed,
        test_edge_cases,
        test_error_curve,
        test_rankdata,
        test_spearman,
        test_permutation_test,
        test_greedy_budget,
        test_greedy_puts_bits_where_sensitive,
        test_noise_floor,
        test_uniform_allocation,
        test_equal_error_arm,
    ]

    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            print(f"\n  [ERROR] {t.__name__} raised: {exc}")
            traceback.print_exc()
            _RESULTS.append((t.__name__, False, str(exc)))

    n_total = len(_RESULTS)
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"  {n_total} checks total, {n_total - n_fail} passed, {n_fail} failed")
    if n_fail:
        print("  failures:")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"    - {name}  {detail}")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
