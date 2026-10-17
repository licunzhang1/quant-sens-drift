"""Diagnostic 4: find the difficulty range where the source domain is not
saturated and the target domain does not collapse.

Why this step is mandatory
--------------------------
A quantization sensitivity analysis requires:
  * the source-domain accuracy **must not be saturated** (otherwise
    Delta loss / Delta acc ~= 0, the sensitivity is pure noise and the rank
    correlation degenerates -- see the NaN the acc metric produces in
    smoke_pipeline)
  * the target-domain accuracy **must not drop to chance level** (otherwise the
    sensitivity ranking is itself noise)

Those two conditions squeeze the "difficulty" into a single range. This script
scans it out.

The key output columns:
  accA   source-domain accuracy       -- want 0.85 ~ 0.97
  dLoss2 largest Delta loss at 2bit in the source domain -- want clearly > 0
  spread spread/magnitude of the source-domain sensitivity -- want real
         resolution rather than all zeros
  accB   target-domain accuracy       -- want 0.35 ~ 0.75
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

    # source-domain per-layer sensitivity at 2bit (only the most aggressive
    # rung, which is enough to judge saturation)
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
          f"{'accA':>7} {'lossA':>8} {'dLoss2':>9} {'spread':>8} {'accB':>7}  verdict")
    print("-" * 96)

    configs = [
        # first check the CFO hypothesis: same (6,14)->(0,8), only CFO changes
        (64, (6, 14), (0, 8), 0.0, 100),
        (64, (6, 14), (0, 8), 0.001, 100),
        (64, (6, 14), (0, 8), 0.005, 100),
        (64, (6, 14), (0, 8), 0.02, 100),
        # CFO=0, SNR shift only, looking for a usable range
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
        verdict = ("OK usable" if (ok_src and ok_tgt)
                   else ("source saturated" if not ok_src else "target unsuitable"))
        print(f"{n_sym:>5} {str(sa):>10} {str(sb):>9} {ep:>4} "
              f"{r['accA']:>7.4f} {r['lossA']:>8.4f} {r['dLoss2_max']:>9.5f} "
              f"{r['dLoss2_spread']:>8.5f} {r['accB']:>7.4f}  {verdict}")
        rows.append((verdict, n_sym, sa, sb, cfo, ep, r))

    good = [x for x in rows if x[0].startswith("OK")]
    print("-" * 96)
    if good:
        print("usable configs:")
        for v, n_sym, sa, sb, cfo, ep, r in good:
            print(f"  n_symbols={n_sym} SNR_A={sa} SNR_B={sb} CFO={cfo} epochs={ep}")
            print(f"    accA={r['accA']:.4f} accB={r['accB']:.4f} "
                  f"dLoss2_max={r['dLoss2_max']:.5f}")
            print(f"    source-domain per-layer Delta loss at 2bit = "
                  f"{np.round(r['sens2'], 5).tolist()}")
    else:
        print("no config satisfies both conditions. Suggestion: lower the "
              "source-domain SNR or use fewer symbols (weaken the source),")
        print("and raise the target-domain SNR appropriately so the target does "
              "not collapse to chance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
