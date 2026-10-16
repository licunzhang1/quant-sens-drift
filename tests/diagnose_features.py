"""诊断 3：累积量特征是否可分？"""

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

from src.data.features import FEATURE_NAMES, amc_features  # noqa: E402
from src.data.synth import ChannelConfig, make_iq_dataset  # noqa: E402
from src.models.numpy_mlp import NumpyMLP  # noqa: E402

MODS = ("bpsk", "qpsk", "8psk", "16qam")


def main():
    cfg = ChannelConfig.from_dict({"snr_db": (10.0, 18.0)})
    N = 4000
    X, y = make_iq_dataset(N, cfg, modulations=MODS, n_symbols=128, sps=4, seed=0)
    F = amc_features(X)
    print(f"特征矩阵: {F.shape}   ({len(FEATURE_NAMES)} 维)")

    print("\n各类的特征均值（看是否有区分度）：")
    print("  {:<18s}".format("feature") + "".join(f"{m:>10s}" for m in MODS) + "   类间/类内")
    for j, name in enumerate(FEATURE_NAMES):
        col = F[:, j]
        means = [float(col[y == c].mean()) for c in range(len(MODS))]
        within = float(np.mean([col[y == c].std() for c in range(len(MODS))]))
        between = float(np.std(means))
        ratio = between / (within + 1e-12)
        print("  {:<18s}".format(name)
              + "".join(f"{v:>10.3f}" for v in means)
              + f"   {ratio:>6.2f}")

    # 用同一批数据做**严格的** train/test 划分
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_tr = int(N * 0.75)
    tr, te = perm[:n_tr], perm[n_tr:]

    mu = F[tr].mean(axis=0)
    sd = F[tr].std(axis=0) + 1e-9
    Fn = (F - mu) / sd

    dims = [Fn.shape[1], 32, 16, len(MODS)]
    m = NumpyMLP(dims, seed=0)
    m.train(Fn[tr], y[tr], epochs=60, batch_size=128, lr=3e-3, weight_decay=1e-4, seed=0)
    tr_acc = m.accuracy(Fn[tr], y[tr])
    te_acc = m.accuracy(Fn[te], y[te])
    print(f"\n严格留出划分（同一数据集 75/25）：train={tr_acc:.4f}  test={te_acc:.4f}")
    print(f"参数量 = {sum(v.size for v in m.params.values())}")

    # 换一个全新生成的数据集（真正检验跨数据集泛化）
    X2, y2 = make_iq_dataset(1500, cfg, modulations=MODS, n_symbols=128, sps=4, seed=999)
    F2 = (amc_features(X2) - mu) / sd
    print(f"新生成数据集（不同种子）      ：test={m.accuracy(F2, y2):.4f}")

    # 逐类
    pred = np.argmax(m.forward(F2), axis=1)
    for c, mod in enumerate(MODS):
        mask = y2 == c
        print(f"    {mod:<6s} acc={np.mean(pred[mask] == c):.3f}  (n={int(mask.sum())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
