"""诊断 2：为什么同域测试准确率卡在 0.50？

假设
----
* H1：符号数/过采样率太低（sps=2, 32 symbols），高阶调制不可分
* H2：标签与波形事实上独立（生成器有 bug）
* H3：accuracy/模型实现有 bug（训练集自身切一半也测不准）

做法：在真实配置附近扫 sps 与符号数，并打印**逐类**测试准确率与类别分布。
若是 H1，提高 sps/符号数应当改善；若是 H2/H3，无论怎么调都不会改善。
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

from src.data.synth import ChannelConfig, make_iq_dataset  # noqa: E402
from src.models.numpy_mlp import NumpyMLP  # noqa: E402

MODS = ("bpsk", "qpsk", "8psk", "16qam")


def per_class_acc(y_true, y_pred, n_classes=4):
    out = []
    for c in range(n_classes):
        m = y_true == c
        out.append(float(np.mean(y_pred[m] == c)) if m.any() else float("nan"))
    return out


def run(n_symbols, sps, n_train=6000, n_test=1500, hidden=(64, 32), epochs=40, seed=0):
    cfg = ChannelConfig.from_dict({"snr_db": (10.0, 18.0)})
    Xtr, ytr = make_iq_dataset(n_train, cfg, modulations=MODS,
                               n_symbols=n_symbols, sps=sps, seed=seed)
    Xte, yte = make_iq_dataset(n_test, cfg, modulations=MODS,
                               n_symbols=n_symbols, sps=sps, seed=seed + 7)

    dims = [Xtr.shape[1] * Xtr.shape[2]] + list(hidden) + [len(MODS)]
    m = NumpyMLP(dims, seed=seed)
    n_par = sum(v.size for v in m.params.values())
    m.train(Xtr, ytr, epochs=epochs, batch_size=128, lr=2e-3,
            weight_decay=1e-3, seed=seed)

    # H3 检查：把训练集切一半当"测试"，若也测不准就是实现 bug
    half = n_train // 2
    acc_self = m.accuracy(Xtr[half:], ytr[half:])
    pred = np.argmax(m.forward(Xte), axis=1)

    return {
        "n_symbols": n_symbols, "sps": sps, "L": Xtr.shape[-1], "params": n_par,
        "train_acc": m.accuracy(Xtr, ytr),
        "self_acc": acc_self,
        "test_acc": float(np.mean(pred == yte)),
        "per_class": per_class_acc(yte, pred),
        "pred_hist": np.bincount(pred, minlength=len(MODS)).tolist(),
        "true_hist": np.bincount(ytr, minlength=len(MODS)).tolist(),
    }


def main():
    print("H3 前置检查：类别分布（训练集）")
    cfg = ChannelConfig.from_dict({"snr_db": (10.0, 18.0)})
    _, y = make_iq_dataset(4000, cfg, modulations=MODS, n_symbols=32, sps=2, seed=0)
    print("  true:", np.bincount(y, minlength=4).tolist(), " (应接近均匀)")

    print("\n扫描 sps 与符号数：")
    print(f"{'n_sym':>6} {'sps':>4} {'L':>6} {'params':>8} {'train':>7} "
          f"{'self':>7} {'test':>7}   逐类测试准确率")
    print("-" * 92)

    for n_symbols, sps in [(32, 2), (32, 4), (64, 4), (128, 4), (128, 8), (256, 4)]:
        r = run(n_symbols, sps)
        pc = " ".join(f"{v:.2f}" for v in r["per_class"])
        print(f"{r['n_symbols']:>6} {r['sps']:>4} {r['L']:>6} {r['params']:>8} "
              f"{r['train_acc']:>7.4f} {r['self_acc']:>7.4f} {r['test_acc']:>7.4f}   {pc}")

    print("\n预测分布 vs 真实分布（最后一组）")
    r = run(256, 4)
    print("  pred:", r["pred_hist"])
    print("  true:", r["true_hist"])
    print("  逐类:", np.round(r["per_class"], 3).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
