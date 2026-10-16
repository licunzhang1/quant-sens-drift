"""端到端冒烟测试：用纯 numpy 参考模型跑通整条实验管线。

它验证的链路（与真实 torch 实验完全同构）：

    合成 IQ（Domain A / Domain B）
      → 经典 AMC 累积量特征
      → 训练参考模型
      → 测两域指标（看域间隙）
      → 逐层选择性量化 → 敏感性矩阵（loss 口径 + acc 口径）
      → Spearman ρ（含置换检验）
      → 等重构误差三臂
      → 落盘结果

跑法：

    python tests/smoke_pipeline.py
    python tests/smoke_pipeline.py --raw-iq    # 复现"原始 IQ 喂 MLP 会失败"

======================================================================
本脚本记录的两个实测发现（真实实验必须注意）
======================================================================

【发现 1】原始 IQ 直接喂 MLP 不行 —— 是表示问题，不是数据问题
------------------------------------------------------------------
训练准确率 1.0、测试 0.50；逐类看只有 BPSK 能分（0.98），
QPSK/8PSK/16QAM 全在 0.33。
原因：调制方式的信息藏在样本的**分布**里，而不是某几个采样点上，
普通 MLP 学不出来。换成经典累积量特征后，同域留出与全新数据集都是 1.0000。
→ 见 tests/diagnose_data.py 与 tests/diagnose_features.py

【发现 2】准确率会饱和，**必须用 loss 作为敏感性的主口径**
------------------------------------------------------------------
当同域准确率已经到 1.0000 时，把某一层量化到 2 bit 仍然可能是 1.0000，
于是 Δacc ≡ 0，敏感性向量全为 0，秩相关直接退化成 NaN。
准确率是**有饱和上限的离散量**，在量化敏感性研究里信息量不足。
→ 所以本脚本同时算 loss 与 acc 两套敏感性，**ρ 用 loss 口径**，
  并把两套并排打印，让这个差异可见。

真实实验配置（configs/base.yaml）里 ``sensitivity.metric`` 默认已设为 ``loss``。

【发现 3】表示方式决定了**哪种域偏移是致命的**
------------------------------------------------------------------
固定同一个 SNR 偏移 (6,14)→(0,8) dB，只改载波频偏 CFO：

    CFO=0.000 → 目标域 acc 0.6875
    CFO=0.001 → 目标域 acc 0.4288
    CFO=0.005 → 目标域 acc 0.3300
    CFO=0.020 → 目标域 acc 0.3137   ← 掉到随机水平

原因是高阶累积量**不是 CFO 不变的**：``E[z^4]`` 在 CFO 下带上 ``exp(j8πfk)``
的相位，L=512、f=0.02 时相位转过 82 圈，累积量直接抵消归零。

这不是 bug，而是一个值得写进报告的现象：**"域偏移的强度"不是绝对量，
它取决于输入表示**。同一个 CFO 对累积量特征是灾难，对能学到相位不变性的
CNN 可能只是轻微退化。本项目的因子分解协议正好能刻画这一点。
→ 见 tests/diagnose_regime.py

因此冒烟测试默认只用 **SNR 偏移**（CFO=0），以便在"源域不饱和、
目标域不退化"的可用区间内验证链路。
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
    ap = argparse.ArgumentParser(description="quant-sens 端到端冒烟测试（纯 numpy）")
    ap.add_argument("--raw-iq", action="store_true",
                    help="直接用原始 IQ 喂 MLP（预期失败，用于复现表示问题）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="results/smoke")
    args = ap.parse_args()

    n_train, n_test, n_sym, sps, epochs = 3000, 1000, 64, 4, 100
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------------ 数据
    banner("步骤 1/6  生成合成 IQ（Domain A 与 Domain B）")
    # 难度取自 tests/diagnose_regime.py 扫出的可用区间：
    # 源域不饱和（acc≈0.95）且目标域不退化（acc≈0.56）。
    dom_a = ChannelConfig.from_dict({"snr_db": (4.0, 12.0)})
    # 只用 SNR 偏移。CFO 对累积量特征是灾难性的（见文件头【发现 3】），
    # 会让目标域直接掉到随机水平，敏感性排序随之变成噪声。
    dom_b = ChannelConfig.from_dict({"snr_db": (-2.0, 6.0)})

    t0 = time.time()
    Xa, ya = make_iq_dataset(n_train, dom_a, modulations=MODS, n_symbols=n_sym, sps=sps,
                             seed=args.seed)
    Xa_te, ya_te = make_iq_dataset(n_test, dom_a, modulations=MODS, n_symbols=n_sym, sps=sps,
                                   seed=args.seed + 1000)
    Xb_te, yb_te = make_iq_dataset(n_test, dom_b, modulations=MODS, n_symbols=n_sym, sps=sps,
                                   seed=args.seed + 2000)
    print(f"  Domain A 训练 : {Xa.shape}   {dom_a.describe()}")
    print(f"  Domain A 测试 : {Xa_te.shape}")
    print(f"  Domain B 测试 : {Xb_te.shape}   {dom_b.describe()}")
    print(f"  用时 {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ 表示
    banner("步骤 2/6  输入表示")
    if args.raw_iq:
        Fa, Fa_te, Fb_te = (x.reshape(x.shape[0], -1) for x in (Xa, Xa_te, Xb_te))
        repr_name = "raw_iq"
        print(f"  原始 IQ 展平: {Fa.shape[1]} 维")
        print("  预期：训练准确率很高、同域测试接近随机 —— 表示不含分布性信息。")
    else:
        Fa, Fa_te, Fb_te = amc_features(Xa), amc_features(Xa_te), amc_features(Xb_te)
        repr_name = "amc_cumulants"
        print(f"  经典 AMC 累积量特征: {Fa.shape[1]} 维")
        print(f"  {', '.join(FEATURE_NAMES)}")

    mu, sd = Fa.mean(axis=0), Fa.std(axis=0) + 1e-9
    Fa, Fa_te, Fb_te = [(F - mu) / sd for F in (Fa, Fa_te, Fb_te)]
    print("  按 Domain A 训练集统计量标准化（不用测试集信息）")

    # ------------------------------------------------------------------ 训练
    banner("步骤 3/6  在 Domain A 上训练参考模型")
    in_dim = Fa.shape[1]
    if args.raw_iq:
        hidden = (128, 64)
    else:
        # 刻意做深一点：层数太少时秩相关几乎没有统计功效
        hidden = (48, 48, 40, 40, 32, 32, 24)
    dims = [in_dim, *hidden, len(MODS)]
    model = NumpyMLP(dims, seed=args.seed)
    n_params = sum(v.size for v in model.params.values())
    print(f"  结构 dims = {dims}")
    print(f"  参数量   = {n_params:,}    权重层数 = {len(model.weight_layer_names())}")
    print("  训练中 ...")
    t0 = time.time()
    hist = model.train(Fa, ya, epochs=epochs, batch_size=128, lr=3e-3,
                       weight_decay=1e-4, seed=args.seed, verbose=False)
    print(f"  用时 {time.time() - t0:.1f}s   loss {hist[0]:.4f} → {hist[-1]:.4f}")

    # ------------------------------------------------------------------ 域间隙
    banner("步骤 4/6  两域指标（域间隙）")
    acc_a_tr = model.accuracy(Fa, ya)
    acc_a = model.accuracy(Fa_te, ya_te)
    acc_b = model.accuracy(Fb_te, yb_te)
    loss_a = model.loss(Fa_te, ya_te)
    loss_b = model.loss(Fb_te, yb_te)
    print(f"  Domain A 训练集      : acc={acc_a_tr:.4f}")
    print(f"  Domain A 测试集 (同域): acc={acc_a:.4f}   loss={loss_a:.4f}")
    print(f"  Domain B 测试集 (目标): acc={acc_b:.4f}   loss={loss_b:.4f}")
    print(f"  同域泛化间隙 = {acc_a_tr - acc_a:+.4f}   （应接近 0）")
    print(f"  跨域掉点     = {acc_a - acc_b:+.4f}    loss 上升 = {loss_b - loss_a:+.4f}")
    if acc_a_tr - acc_a > 0.05:
        print("  [!] 同域间隙偏大：域间隙会与泛化失败混淆。")
    if acc_b < 0.30:
        print("  [!] 目标域接近随机水平：敏感性排序本身就是噪声，应减弱偏移。")
    if acc_a > 0.999:
        print("  [!] 源域准确率饱和到 1.0：Δacc 口径会退化成全 0（见下方对比）。")

    # ------------------------------------------------------------------ 敏感性
    banner("步骤 5/6  逐层敏感性矩阵（loss 口径 + acc 口径）")
    layer_names = model.weight_layer_names()
    ev_a = lambda: model.loss(Fa_te, ya_te)      # noqa: E731
    ev_b = lambda: model.loss(Fb_te, yb_te)      # noqa: E731
    ac_a = lambda: model.accuracy(Fa_te, ya_te)  # noqa: E731
    ac_b = lambda: model.accuracy(Fb_te, yb_te)  # noqa: E731
    ql = lambda name, b: quantized_layer(model, name, b, per_channel=True)  # noqa: E731

    print("\n  [Domain A] loss 口径")
    sens_a = compute_sensitivity_matrix(layer_names, BITS_LADDER, evaluate=ev_a,
                                        quantize_layer=ql, metric="loss",
                                        higher_is_better=False, domain="A",
                                        backbone="numpy_mlp")
    print("\n  [Domain B] loss 口径")
    sens_b = compute_sensitivity_matrix(layer_names, BITS_LADDER, evaluate=ev_b,
                                        quantize_layer=ql, metric="loss",
                                        higher_is_better=False, domain="B",
                                        backbone="numpy_mlp")
    print("\n  [Domain A] acc 口径（用于展示饱和问题）")
    accsens_a = compute_sensitivity_matrix(layer_names, BITS_LADDER, evaluate=ac_a,
                                           quantize_layer=ql, metric="acc",
                                           higher_is_better=True, domain="A",
                                           backbone="numpy_mlp")
    print("\n  [Domain B] acc 口径")
    accsens_b = compute_sensitivity_matrix(layer_names, BITS_LADDER, evaluate=ac_b,
                                           quantize_layer=ql, metric="acc",
                                           higher_is_better=True, domain="B",
                                           backbone="numpy_mlp")

    print_matrix("Δloss（越大越敏感，主口径）", layer_names, BITS_LADDER, sens_a.delta)
    print_matrix("Δloss（Domain B）", layer_names, BITS_LADDER, sens_b.delta)
    print_matrix("Δacc（Domain A，注意饱和）", layer_names, BITS_LADDER, accsens_a.delta,
                 "{:>12.4f}")
    print_matrix("Δacc（Domain B）", layer_names, BITS_LADDER, accsens_b.delta, "{:>12.4f}")

    # ------------------------------------------------------------------ rho
    banner("步骤 6/6  秩稳定性 ρ（核心量）")
    sa = sens_a.sensitivity_at(REFERENCE_BITS)
    sb = sens_b.sensitivity_at(REFERENCE_BITS)
    ta = accsens_a.sensitivity_at(REFERENCE_BITS)
    tb = accsens_b.sensitivity_at(REFERENCE_BITS)
    print(f"  参考位宽 = {REFERENCE_BITS} bit   层数 n = {len(layer_names)}")
    print(f"  Domain A 敏感性(loss): {np.round(sa, 5).tolist()}")
    print(f"  Domain B 敏感性(loss): {np.round(sb, 5).tolist()}")

    res = analyse_pair(sa, sb, n_permutations=20000, seed=args.seed)
    print(f"\n  === loss 口径（主）===")
    print(f"  ρ (Spearman) = {res.rho:+.4f}    p = {res.p_value:.4g}")
    print(f"  95% CI = [{res.ci_low:+.3f}, {res.ci_high:+.3f}]   Kendall τ-b = {res.tau_b:+.4f}")
    print(f"  结论: {res.verdict()}")

    res_acc = analyse_pair(ta, tb, n_permutations=20000, seed=args.seed)
    print(f"\n  === acc 口径（对照，展示饱和的后果）===")
    print(f"  ρ (Spearman) = {res_acc.rho:+.4f}    p = {res_acc.p_value:.4g}")
    print(f"  结论: {res_acc.verdict()}")
    print(f"  A 域 acc 敏感性唯一值个数 = {int(np.unique(ta).size)} / {ta.size}")

    # ------------------------------------------------------------------ 等误差臂
    banner("附 1：等重构误差臂（三臂对比）")
    curves = {n: quant_error_curve(model.params[n], BITS_LADDER, True, 0)
              for n in layer_names}
    err = np.array([[curves[n][b]["nmse"] for b in BITS_LADDER] for n in layer_names])
    print_matrix("逐层 NMSE ε_l(b)", layer_names, BITS_LADDER, err)

    costs = model.layer_costs()
    arm = run_equal_error_arm(sa, sb, err, BITS_LADDER, budget_ratio=0.6,
                              layer_costs=costs, layer_names=layer_names,
                              source_domain="A", target_domain="B")
    print(f"\n  预算占比 = {arm.budget_ratio}   总预算 = {arm.budget_total:.1f}")
    print("    {:<20s}{:<26s}{:>16s}".format("arm", "bits", "pred_damage"))
    for tag, alloc in (("uniform", arm.uniform),
                       ("agnostic (源域A)", arm.agnostic),
                       ("oracle (目标域B)", arm.oracle)):
        print("    {:<20s}{:<26s}{:>16.8f}".format(
            tag, str(alloc.bits.tolist()), alloc.predicted_damage))
    print(f"\n  敏感性引导的收益 (uniform - agnostic) = {arm.gain_from_sensitivity:+.8f}")
    print(f"  **不知道部署域的代价** (agnostic - oracle) = {arm.cost_of_domain_ignorance:+.8f}")
    print(f"  相对代价 = {arm.relative_cost_of_ignorance:+.1%}")
    print(f"  排序是否改变决策: {arm.ranking_flipped}")
    print(f"  结论: {arm.verdict()}")

    # ------------------------------------------------------------------ 代理量
    banner("附 2：代理量能否预测敏感性（加分项）")
    tensors = {n: model.params[n] for n in layer_names}
    prox, _ = proxy_matrix(tensors, bits=REFERENCE_BITS)
    rows = correlate_with_sensitivity(prox, sb, n_permutations=20000, seed=args.seed)
    for r in rows[:6]:
        flag = " *" if r["significant"] else ""
        print(f"    {r['proxy']:<14s} ρ={r['rho']:+.4f}  p={r['p_value']:.4f}{flag}")
    print("  （* 表示 p<0.05。层数个位数时，单看某个代理量的显著性要谨慎。）")

    # ------------------------------------------------------------------ 落盘
    banner("结果落盘")
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
    print(f"  已写入 {p_json}")
    print(f"  已写入 {outdir / f'sensitivity_{tag}.npz'}")

    banner("总结")
    print(f"  管线端到端跑通，用时 {time.time() - t_start:.1f}s")
    print(f"  表示                    : {repr_name}")
    print(f"  同域泛化间隙            : {acc_a_tr - acc_a:+.4f}")
    print(f"  跨域掉点 (Δacc)         : {acc_a - acc_b:+.4f}")
    print(f"  秩稳定性 ρ (loss 口径)  : {res.rho:+.4f}  (p={res.p_value:.4g})")
    print(f"  秩稳定性 ρ (acc  口径)  : {res_acc.rho:+.4f}  (p={res_acc.p_value:.4g})")
    print(f"  域无知代价              : {arm.cost_of_domain_ignorance:+.8f}")
    print()
    print("  这个脚本只验证'链路通不通'，不产生科学结论：")
    print("  · 参考模型不是真实骨干（真实实验用 CLDNN / 1D-ResNet）")
    print("  · 只用了两类偏移、单一骨干，ρ 的估计方差很大")
    return 0


if __name__ == "__main__":
    sys.exit(main())
