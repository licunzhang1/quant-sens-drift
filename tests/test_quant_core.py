"""核心量化数学与秩相关分析的单元测试。

只依赖 numpy，**不需要 torch**。可以直接跑：

    python tests/test_quant_core.py

也可以用 pytest 跑。
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
# 1. 量化区间
# ==========================================================================
def test_qrange() -> None:
    print("\n[1] 量化区间")
    check("qrange(8, sym) == (-127, 127)", qrange(8, True) == (-127, 127), str(qrange(8, True)))
    check("qrange(8, asym) == (0, 255)", qrange(8, False) == (0, 255), str(qrange(8, False)))
    check("qrange(2, sym) == (-1, 1)", qrange(2, True) == (-1, 1), str(qrange(2, True)))
    check("qrange(4, asym) == (0, 15)", qrange(4, False) == (0, 15), str(qrange(4, False)))

    raised = False
    try:
        qrange(1, True)
    except ValueError:
        raised = True
    check("qrange(1, sym) 抛 ValueError（qmax 会是 0）", raised)

    raised = False
    try:
        qrange(0, False)
    except ValueError:
        raised = True
    check("qrange(0, asym) 抛 ValueError", raised)


# ==========================================================================
# 2. 量化正确性
# ==========================================================================
def test_exact_quantization() -> None:
    print("\n[2] 可精确表示的张量应零误差")
    x = np.array([-1.0, 0.0, 1.0])
    xq, scale, zero = quantize_dequantize(x, 2, symmetric=True)
    check("x=[-1,0,1], 2bit 对称 → 无误差", approx(np.max(np.abs(x - xq)), 0.0),
          f"max_err={np.max(np.abs(x - xq)):.3e}, scale={scale}, zero={zero}")

    err = reconstruction_error(x, xq)
    check("重构误差 nmse == 0", approx(err["nmse"], 0.0, 1e-12), str(err))

    # 整数量化码必须落在区间内
    rng = np.random.default_rng(0)
    y = rng.standard_normal(1000) * 5
    for bits in (2, 3, 4, 6, 8):
        qmin, qmax = qrange(bits, True)
        q = quantize(y, bits, symmetric=True)
        ok = (q.min() >= qmin - 1e-9) and (q.max() <= qmax + 1e-9)
        check(f"  量化码在区间内 (bits={bits})", ok, f"[{q.min()},{q.max()}] vs [{qmin},{qmax}]")


def test_error_monotone_in_bits() -> None:
    print("\n[3] 重构误差随位宽单调下降")
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 32)) * 2.0
    ladder = [2, 3, 4, 6, 8]
    errs = [reconstruction_error(x, quantize_dequantize(x, b, True)[0])["nmse"] for b in ladder]
    print("      nmse:", "  ".join(f"{b}bit={e:.4f}" for b, e in zip(ladder, errs)))
    mono = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    check("nmse 严格单调下降", mono, str(errs))
    check("8bit 的 nmse 远小于 2bit", errs[-1] < errs[0] / 10, f"{errs[-1]:.5f} vs {errs[0]:.5f}")


def test_per_channel_vs_per_tensor() -> None:
    print("\n[4] per-channel 不劣于 per-tensor（通道尺度差异大时明显更好）")
    rng = np.random.default_rng(2)
    # 构造通道尺度差异极大的权重矩阵（真实网络里很常见）
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

    # 全局 NMSE 是按能量加权的，大尺度通道会主导数字，
    # 掩盖"小尺度通道被量化摧毁"这件事。用逐通道无权 NMSE 才看得清。
    xq_pt = quantize_dequantize(W, 4, True, None)[0]
    xq_pc = quantize_dequantize(W, 4, True, 0)[0]
    cw_pt = channel_wise_nmse(W, xq_pt, axis=0)
    cw_pc = channel_wise_nmse(W, xq_pc, axis=0)
    print(f"      逐通道无权 NMSE @4bit: per-tensor={cw_pt:.4f}  per-channel={cw_pc:.4f}")
    check("  逐通道无权 NMSE：per-channel 至少好 10 倍", cw_pc * 10 < cw_pt,
          f"pc={cw_pc:.6f} pt={cw_pt:.6f} 比值={cw_pt / max(cw_pc, 1e-15):.1f}x")

    # 而且 per-channel 下每个通道都应被合理表示（不该有通道被彻底摧毁）
    ax = 1
    per_ch = np.mean((W - xq_pc) ** 2, axis=ax) / (np.mean(W ** 2, axis=ax) + 1e-12)
    check("  4bit per-channel 下没有通道被完全摧毁",
          bool(np.all(per_ch < 0.5)), str(np.round(per_ch, 4).tolist()))


def test_asymmetric_for_skewed() -> None:
    print("\n[5] 单边分布上非对称优于对称")
    rng = np.random.default_rng(3)
    x = rng.exponential(scale=1.0, size=20000)  # 全正、长尾
    for bits in (3, 4, 6):
        e_sym = reconstruction_error(x, quantize_dequantize(x, bits, True)[0])["nmse"]
        e_asym = reconstruction_error(x, quantize_dequantize(x, bits, False)[0])["nmse"]
        check(
            f"  bits={bits}: 非对称 <= 对称",
            e_asym <= e_sym + 1e-12,
            f"sym={e_sym:.6f} asym={e_asym:.6f}",
        )


def test_edge_cases() -> None:
    print("\n[6] 边界情况")
    z = np.zeros((8, 8))
    xq, scale, zero = quantize_dequantize(z, 4, True)
    check("全零张量不产生 NaN/Inf", np.all(np.isfinite(xq)), str(xq.flat[:3]))

    const = np.full((4, 4), 3.5)
    xq2, _, _ = quantize_dequantize(const, 4, True)
    check("常量张量不产生 NaN/Inf", np.all(np.isfinite(xq2)))

    g = np.linspace(-1, 1, 1000).reshape(10, 100)
    scale_s, zero_s = quant_params(g, 4, True, None)
    scale_c, zero_c = quant_params(g, 4, True, 0)
    check("per-channel 的 scale 形状可广播", scale_c.shape == (10, 1), str(scale_c.shape))
    check("per-tensor 的 scale 是标量状", np.size(scale_s) == 1, str(np.shape(scale_s)))
    check("per-channel 的 zero_point 形状一致", zero_c.shape == (10, 1), str(zero_c.shape))
    check("对称量化 zero_point 恒为 0", float(np.max(np.abs(zero_s))) == 0.0
          and float(np.max(np.abs(zero_c))) == 0.0)

    raised = False
    try:
        quant_params(np.array([]), 4)
    except ValueError:
        raised = True
    check("空数组抛 ValueError", raised)


def test_error_curve() -> None:
    print("\n[7] 误差曲线接口")
    rng = np.random.default_rng(4)
    x = rng.standard_normal((16, 32))
    curve = quant_error_curve(x, [2, 4, 8])
    check("返回 3 个位宽", len(curve) == 3, str(list(curve.keys())))
    check("每个条目含 nmse", all("nmse" in v for v in curve.values()))
    check("nmse 随位宽下降", curve[8]["nmse"] < curve[4]["nmse"] < curve[2]["nmse"],
          str({k: round(v["nmse"], 5) for k, v in curve.items()}))

    # 位宽 < 2 且对称 → 跳过而不是崩溃
    curve2 = quant_error_curve(x, [1, 2, 4], symmetric=True)
    check("对称模式下 bits=1 被跳过", 1 not in curve2, str(list(curve2.keys())))


# ==========================================================================
# 8. 秩相关
# ==========================================================================
def test_rankdata() -> None:
    print("\n[8] 平均秩")
    r = rankdata_average([3.0, 1.0, 2.0])
    check("无并列: [3,1,2] → [3,1,2]", np.allclose(r, [3, 1, 2]), str(r))

    r = rankdata_average([10.0, 20.0, 20.0, 30.0])
    check("有并列: 取平均秩 → [1, 2.5, 2.5, 4]", np.allclose(r, [1, 2.5, 2.5, 4]), str(r))

    r = rankdata_average([5.0, 5.0, 5.0])
    check("全并列 → 全为 2", np.allclose(r, [2, 2, 2]), str(r))

    rng = np.random.default_rng(5)
    a = rng.standard_normal(50)
    check("秩与排序一致", np.array_equal(np.argsort(rankdata_average(a)), np.argsort(a)))


def test_spearman() -> None:
    print("\n[9] Spearman")
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    check("严格单调递增 → ρ=1", approx(spearman(a, a ** 3), 1.0, 1e-9), str(spearman(a, a ** 3)))
    check("严格单调递减 → ρ=-1", approx(spearman(a, -a), -1.0, 1e-9), str(spearman(a, -a)))

    b = a + 100.0
    check("平移不变 → ρ=1", approx(spearman(a, b), 1.0, 1e-9), str(spearman(a, b)))

    c = np.exp(a)
    check("非线性但单调 → ρ=1", approx(spearman(a, c), 1.0, 1e-9), str(spearman(a, c)))

    # 与 scipy 交叉验证（如果装了）
    try:
        from scipy.stats import spearmanr  # type: ignore

        rng = np.random.default_rng(6)
        for _ in range(5):
            u = rng.standard_normal(20)
            v = rng.standard_normal(20)
            ours = spearman(u, v)
            theirs = float(spearmanr(u, v).statistic)
            if not approx(ours, theirs, 1e-9):
                check("与 scipy.stats.spearmanr 一致", False, f"{ours} vs {theirs}")
                return
        check("与 scipy.stats.spearmanr 一致（5 组随机样本）", True)
    except ImportError:
        check("与 scipy 交叉验证（scipy 未安装，跳过）", True, "skipped")


def test_permutation_test() -> None:
    print("\n[10] 置换检验")
    rng = np.random.default_rng(7)

    x = rng.standard_normal(12)
    # 强相关
    rho, p = permutation_test(x, x * 2 + 1, n_permutations=2000, seed=1)
    check("完全相关 → ρ≈1, p 很小", approx(rho, 1.0, 1e-9) and p < 0.01, f"rho={rho}, p={p}")

    # 完全独立：跑多次，看 p 值是否大致均匀（不应该总是小）
    ps = []
    for s in range(20):
        y = rng.standard_normal(12)
        _, pv = permutation_test(x, y, n_permutations=500, seed=100 + s)
        ps.append(pv)
    frac_small = float(np.mean(np.array(ps) < 0.05))
    check(
        "独立数据 → 拒绝率接近名义 5%（不严重高估显著性）",
        frac_small <= 0.30,
        f"20 次里 {int(frac_small * 20)} 次 p<0.05，p 值={np.round(ps, 3).tolist()}",
    )

    lo, hi = bootstrap_ci(x, x * 2 + 1, n_boot=500, seed=2)
    check("完全相关时 CI 收敛到 1", lo > 0.9 and hi <= 1.0 + 1e-9, f"[{lo:.3f}, {hi:.3f}]")

    check("Kendall τ-b 完全一致 → 1", approx(kendall_tau_b(x, x * 2 + 1), 1.0, 1e-9))
    check("Kendall τ-b 完全反向 → -1", approx(kendall_tau_b(x, -x), -1.0, 1e-9))


# ==========================================================================
# 11. 等重构误差臂
# ==========================================================================
def _fake_err_curves(n_layers: int, ladder=(2, 3, 4, 6, 8), seed=0):
    """构造形状合理的误差曲线：随位宽单调下降，层间有差异。"""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 0.5, size=(n_layers, 1))
    shape = np.array([[1.0, 0.6, 0.35, 0.15, 0.05]], dtype=np.float64)
    return base * shape


def test_greedy_budget() -> None:
    print("\n[11] 贪心分配满足预算")
    E = _fake_err_curves(8, seed=11)
    s = np.linspace(0.1, 1.0, 8)
    costs = np.ones(8)

    for ratio in (0.3, 0.5, 0.8, 1.0):
        alloc = greedy_allocate(s, E, [2, 3, 4, 6, 8], budget_ratio=ratio, layer_costs=costs)
        check(
            f"  ratio={ratio}: 预算未被突破",
            alloc.budget_used <= alloc.budget_total + 1e-6,
            f"used={alloc.budget_used:.2f} total={alloc.budget_total:.2f}",
        )
        check(
            f"  ratio={ratio}: 位宽都在阶梯上",
            set(np.unique(alloc.bits)).issubset({2, 3, 4, 6, 8}),
            str(alloc.bits.tolist()),
        )

    # 预算越大，平均位宽越高
    means = [
        greedy_allocate(s, E, [2, 3, 4, 6, 8], budget_ratio=r, layer_costs=costs).bits.mean()
        for r in (0.2, 0.5, 0.9)
    ]
    check("预算越大平均位宽越高", means[0] <= means[1] <= means[2], str(means))


def test_greedy_puts_bits_where_sensitive() -> None:
    print("\n[12] 贪心把位宽优先给敏感层")
    E = _fake_err_curves(6, seed=12)
    costs = np.ones(6)
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0])  # 只有最后一层敏感

    alloc = greedy_allocate(s, E, [2, 3, 4, 6, 8], budget_ratio=0.5, layer_costs=costs)
    check("敏感层拿到最高位宽", alloc.bits[-1] == alloc.bits.max(), str(alloc.bits.tolist()))
    check("不敏感层停在最低位宽", np.all(alloc.bits[:-1] == 2), str(alloc.bits.tolist()))

    # 敏感性全相等时，分配应当接近均匀
    s_eq = np.ones(6)
    alloc_eq = greedy_allocate(s_eq, E, [2, 3, 4, 6, 8], budget_ratio=0.6, layer_costs=costs)
    spread = int(alloc_eq.bits.max() - alloc_eq.bits.min())
    check("敏感性相等时分配较均匀（最大最小位宽差 <= 2 档）", spread <= 4,
          str(alloc_eq.bits.tolist()))


def test_noise_floor() -> None:
    print("\n[13] 负敏感性被截断（避免把位宽从噪声层挪走）")
    E = _fake_err_curves(5, seed=13)
    s = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])  # 全是"量化反而变好"的噪声
    alloc = greedy_allocate(s, E, [2, 3, 4, 6, 8], budget_ratio=0.9, layer_costs=np.ones(5))
    check("全负敏感性 → 全部停在最低位宽（不浪费预算）",
          np.all(alloc.bits == 2), str(alloc.bits.tolist()))

    s_mix = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    alloc_mix = greedy_allocate(s_mix, E, [2, 3, 4, 6, 8], budget_ratio=0.6, layer_costs=np.ones(5))
    check("负/零敏感性的层不拿额外位宽",
          np.all(alloc_mix.bits[:2] == 2), str(alloc_mix.bits.tolist()))


def test_uniform_allocation() -> None:
    print("\n[14] 均匀分配")
    E = _fake_err_curves(4, seed=14)
    s = np.ones(4)
    alloc = uniform_allocate(4, [2, 3, 4, 6, 8], budget_ratio=0.5,
                             layer_costs=np.ones(4), sensitivity=s, err_curves=E)
    check("所有层同位宽", len(set(alloc.bits.tolist())) == 1, str(alloc.bits.tolist()))
    check("预算不超", alloc.budget_used <= alloc.budget_total + 1e-6,
          f"{alloc.budget_used} <= {alloc.budget_total}")


def test_equal_error_arm() -> None:
    print("\n[15] 三臂对比")
    E = _fake_err_curves(10, seed=15)
    ladder = [2, 3, 4, 6, 8]
    costs = np.ones(10)

    # (a) 两个域敏感性完全一致 → agnostic == oracle，不知道域没有代价
    s = np.linspace(0.1, 1.0, 10)
    r_same = run_equal_error_arm(s, s.copy(), E, ladder, 0.6, costs)
    check("敏感性一致时 agnostic == oracle",
          np.array_equal(r_same.agnostic.bits, r_same.oracle.bits),
          f"{r_same.agnostic.bits.tolist()} vs {r_same.oracle.bits.tolist()}")
    check("敏感性一致时 域无知代价 ≈ 0",
          abs(r_same.cost_of_domain_ignorance) < 1e-12,
          str(r_same.cost_of_domain_ignorance))

    # (b) 两个域敏感性完全反转 → 应当产生正的域无知代价
    s_rev = s[::-1].copy()
    r_rev = run_equal_error_arm(s, s_rev, E, ladder, 0.6, costs)
    check("排序反转时 agnostic != oracle",
          not np.array_equal(r_rev.agnostic.bits, r_rev.oracle.bits),
          f"{r_rev.agnostic.bits.tolist()} vs {r_rev.oracle.bits.tolist()}")
    check("排序反转时 域无知代价 > 0",
          r_rev.cost_of_domain_ignorance > 0,
          str(r_rev.cost_of_domain_ignorance))
    check("ranking_flipped 标记正确", r_rev.ranking_flipped is True)
    print(f"      verdict: {r_rev.verdict()}")

    # (c) 预测损伤的一致性
    d = predicted_damage(r_rev.oracle.bits, s_rev, E, ladder)
    check("三臂损伤用目标域敏感性口径一致",
          approx(d, r_rev.oracle.predicted_damage, 1e-9),
          f"{d} vs {r_rev.oracle.predicted_damage}")


# ==========================================================================
def main() -> int:
    print("=" * 74)
    print("  quant-sens 核心单元测试（纯 numpy，不需要 torch）")
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
            print(f"\n  [ERROR] {t.__name__} 抛出异常: {exc}")
            traceback.print_exc()
            _RESULTS.append((t.__name__, False, str(exc)))

    n_total = len(_RESULTS)
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"  合计 {n_total} 项检查，通过 {n_total - n_fail}，失败 {n_fail}")
    if n_fail:
        print("  失败项：")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"    - {name}  {detail}")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
