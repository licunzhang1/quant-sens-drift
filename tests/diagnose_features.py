"""Diagnostic 3: are the cumulant features separable?"""

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
    print(f"feature matrix: {F.shape}   ({len(FEATURE_NAMES)} dims)")

    print("\nper-class feature means (to see whether they discriminate):")
    print("  {:<18s}".format("feature") + "".join(f"{m:>10s}" for m in MODS)
          + "   between/within")
    for j, name in enumerate(FEATURE_NAMES):
        col = F[:, j]
        means = [float(col[y == c].mean()) for c in range(len(MODS))]
        within = float(np.mean([col[y == c].std() for c in range(len(MODS))]))
        between = float(np.std(means))
        ratio = between / (within + 1e-12)
        print("  {:<18s}".format(name)
              + "".join(f"{v:>10.3f}" for v in means)
              + f"   {ratio:>6.2f}")

    # make a **strict** train/test split of the same data
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
    print(f"\nstrict holdout split (same dataset, 75/25): "
          f"train={tr_acc:.4f}  test={te_acc:.4f}")
    print(f"params = {sum(v.size for v in m.params.values())}")

    # switch to a freshly generated dataset (a real test of cross-dataset
    # generalization)
    X2, y2 = make_iq_dataset(1500, cfg, modulations=MODS, n_symbols=128, sps=4, seed=999)
    F2 = (amc_features(X2) - mu) / sd
    print(f"freshly generated dataset (different seed): test={m.accuracy(F2, y2):.4f}")

    # per class
    pred = np.argmax(m.forward(F2), axis=1)
    for c, mod in enumerate(MODS):
        mask = y2 == c
        print(f"    {mod:<6s} acc={np.mean(pred[mask] == c):.3f}  (n={int(mask.sum())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
