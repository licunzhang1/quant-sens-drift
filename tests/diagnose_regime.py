"""诊断 4：找出"源域不饱和、目标域不退化"的难度区间。

为什么必须做这一步
------------------
量化敏感性分析要求：
  * 源域准确率**不能饱和**（否则 Δloss/Δacc ≈ 0，敏感性全是噪声，
    秩相关退化 —— 见 smoke_pipeline 里 acc 口径得到 NaN 的现象）
  * 目标域准确率**不能掉到随机水平**（否则敏感性排序本身就是噪声）

两个条件把"难度"夹在一个区间里。这个脚本把它扫出来。

输出的关键列：
  accA   源域准确率        —— 想要 0.85 ~ 0.97
  dLoss2 源域 2bit 时最大 Δloss —— 想要明显 > 0
  spread 源域敏感性的极差/量级 —— 想要有区分度而不是全零
  accB   目标域准确率      —— 想要 0.35 ~ 0.75
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

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import amc_features  # noqa: E402
from src.data.synth import ChannelConfig, make_iq_dataset  # noqa: E402
from src.models.numpy_mlp import NumpyMLP, quantized_layer  # noqa: E402

MODS = ("bpsk", "qpsk", "8psk", "16qam")
LADDER = [2, 3, 4, 6, 8]


def build(n_symbols, snr_a, snr_b, cfo_b, epochs, n_train=3000, n_test=800, seed=0):
    cfg_a = ChannelConfig.from_dict({"snr_db": snr_a})
    cfg_b = ChannelConfig.from_dict({"snr_db": snr_b, "cfo": cfo_b})
    Xa, ya = make_iq_dataset(n_train, cfg_a, modulations=MODS, n_symbols=n_symbols,
                             sps=4, seed=seed)
    Xat, yat = make_iq_dataset(n_test, cfg_a, modulations=MODS, n_symbols=n_symbols,
                               sps=4, seed=seed + 1000)
    Xbt, ybt = make_iq_dataset(n_test, cfg_b, modulations=MODS, n_symbols=n_symbols,
                               sps=4, seed=seed + 2000)
    Fa, Fat, Fbt = amc_features(Xa), amc_features(Xat), amc_features(Xbt)
    mu, sd = Fa.mean(0), Fa.std(0) + 1e-9
    Fa, Fat, Fbt = [(F - mu) / sd for F in (Fa, Fat, Fbt)]

    dims = [Fa.shape[1], 48, 48, 40, 40, 32, 32, 24, 4]
    m = NumpyMLP(dims, seed=seed)
    m.train(Fa, ya, epochs=epochs, batch_size=128, lr=3e-3, weight_decay=1e-4, seed=seed)

    # 源域 2bit 逐层敏感性（只算最激进的一档，够判断是否饱和）
    base = m.loss(Fat, yat)
    sens2 = []
    for name in m.weight_layer_names():
        with quantized_layer(m, name, 2, per_channel=True):
            sens2.append(m.loss(Fat, yat) - base)
    sens2 = np.array(sens2)

    return {
        "accA": m.accuracy(Fat, yat),
        "lossA": base,
        "accB": m.accuracy(Fbt, ybt),
        "lossB": m.loss(Fbt, ybt),
        "dLoss2_max": float(np.nanmax(sens2)),
        "dLoss2_spread": float(np.nanmax(sens2) - np.nanmin(sens2)),
        "sens2": sens2,
    }


def main():
    print(f"{'n_sym':>5} {'SNR_A':>10} {'SNR_B':>9} {'ep':>4} "
          f"{'accA':>7} {'lossA':>8} {'dLoss2':>9} {'spread':>8} {'accB':>7}  判定")
    print("-" * 96)

    configs = [
        # 先验证 CFO 假设：同为 (6,14)→(0,8)，只改 CFO
        (64, (6, 14), (0, 8), 0.0, 100),
        (64, (6, 14), (0, 8), 0.001, 100),
        (64, (6, 14), (0, 8), 0.005, 100),
        (64, (6, 14), (0, 8), 0.02, 100),
        # CFO=0，只做 SNR 偏移，找可用区间
        (128, (10, 18), (2, 10), 0.0, 60),
        (128, (10, 18), (0, 8), 0.0, 60),
        (64, (10, 18), (2, 10), 0.0, 80),
        (64, (8, 16), (0, 8), 0.0, 80),
        (64, (6, 14), (-2, 6), 0.0, 100),
        (64, (4, 12), (-2, 6), 0.0, 100),
        (32, (8, 16), (0, 8), 0.0, 80),
        (32, (6, 14), (-2, 6), 0.0, 80),
    ]

    rows = []
    for n_sym, sa, sb, cfo, ep in configs:
        r = build(n_sym, sa, sb, cfo, ep)
        ok_src = 0.85 <= r["accA"] <= 0.975 and r["dLoss2_max"] > 1e-3
        ok_tgt = 0.35 <= r["accB"] <= 0.75
        verdict = ("OK 可用" if (ok_src and ok_tgt)
                   else ("源域饱和" if not ok_src else "目标域不合适"))
        print(f"{n_sym:>5} {str(sa):>10} {str(sb):>9} {ep:>4} "
              f"{r['accA']:>7.4f} {r['lossA']:>8.4f} {r['dLoss2_max']:>9.5f} "
              f"{r['dLoss2_spread']:>8.5f} {r['accB']:>7.4f}  {verdict}")
        rows.append((verdict, n_sym, sa, sb, cfo, ep, r))

    good = [x for x in rows if x[0].startswith("OK")]
    print("-" * 96)
    if good:
        print("可用配置：")
        for v, n_sym, sa, sb, cfo, ep, r in good:
            print(f"  n_symbols={n_sym} SNR_A={sa} SNR_B={sb} CFO={cfo} epochs={ep}")
            print(f"    accA={r['accA']:.4f} accB={r['accB']:.4f} "
                  f"dLoss2_max={r['dLoss2_max']:.5f}")
            print(f"    源域 2bit 逐层 Δloss = {np.round(r['sens2'], 5).tolist()}")
    else:
        print("没有配置同时满足条件。建议：降低源域 SNR 或减少符号数（削弱源域），")
        print("同时适当提高目标域 SNR（避免目标域退化到随机）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
