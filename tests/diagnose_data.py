"""Diagnostic 2: why does in-domain test accuracy stall at 0.50?

Hypotheses
----------
* H1: too few symbols / too low an oversampling rate (sps=2, 32 symbols), so the
  higher-order modulations are not separable
* H2: the labels are effectively independent of the waveform (a generator bug)
* H3: a bug in the accuracy/model implementation (even half of the training set
  scores badly)

Method: sweep sps and the symbol count near the real config, and print the
**per-class** test accuracy and the class distribution.
If it is H1, raising sps/symbols should help; if it is H2/H3, nothing will help.
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

    # H3 check: use half the training set as a "test"; if that also scores
    # badly, it is an implementation bug
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
    print("H3 preliminary check: class distribution (training set)")
    cfg = ChannelConfig.from_dict({"snr_db": (10.0, 18.0)})
    _, y = make_iq_dataset(4000, cfg, modulations=MODS, n_symbols=32, sps=2, seed=0)
    print("  true:", np.bincount(y, minlength=4).tolist(), " (should be near uniform)")

    print("\nsweep over sps and symbol count:")
    print(f"{'n_sym':>6} {'sps':>4} {'L':>6} {'params':>8} {'train':>7} "
          f"{'self':>7} {'test':>7}   per-class test accuracy")
    print("-" * 92)

    for n_symbols, sps in [(32, 2), (32, 4), (64, 4), (128, 4), (128, 8), (256, 4)]:
        r = run(n_symbols, sps)
        pc = " ".join(f"{v:.2f}" for v in r["per_class"])
        print(f"{r['n_symbols']:>6} {r['sps']:>4} {r['L']:>6} {r['params']:>8} "
              f"{r['train_acc']:>7.4f} {r['self_acc']:>7.4f} {r['test_acc']:>7.4f}   {pc}")

    print("\npredicted distribution vs true distribution (last setting)")
    r = run(256, 4)
    print("  pred:", r["pred_hist"])
    print("  true:", r["true_hist"])
    print("  per-class:", np.round(r["per_class"], 3).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
