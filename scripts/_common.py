"""Shared helpers for scripts/: config, model building, data generation, evaluation."""

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
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synth import ChannelConfig, make_iq_dataset  # noqa: E402
from src.models.cldnn import CLDNN  # noqa: E402
from src.models.resnet1d import ResNet1D  # noqa: E402
from src.utils import load_config as _load_config  # noqa: E402

MODS_DEFAULT = ("bpsk", "qpsk", "8psk", "16qam")


def load_config(path: str) -> Dict:
    return _load_config(path)


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def build_model(cfg: Dict, n_classes: Optional[int] = None) -> nn.Module:
    m = cfg["model"]
    name = m.get("name", "cldnn")
    k = int(n_classes if n_classes is not None else m.get("n_classes", 4))
    if name == "cldnn":
        sub = dict(m.get("cldnn") or {})
        if "conv_channels" in sub:
            sub["conv_channels"] = tuple(sub["conv_channels"])
        return CLDNN(n_classes=k, **sub)
    if name == "resnet1d":
        sub = dict(m.get("resnet1d") or {})
        if "n_blocks" in sub:
            sub["n_blocks"] = tuple(sub["n_blocks"])
        return ResNet1D(n_classes=k, **sub)
    raise ValueError(f"unknown model: {name}")


def make_domain(
    cfg: Dict, domain: str, n: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one domain worth of data from the config. ``domain`` is in {'a', 'b', or a key of shifts}."""
    d = cfg["data"]
    if domain == "a":
        ch = d["domain_a"]
    elif domain in (d.get("shifts") or {}):
        ch = d["shifts"][domain]
    else:
        raise KeyError(f"unknown domain {domain!r}; available: a, {list((d.get('shifts') or {}).keys())}")

    return make_iq_dataset(
        n,
        ChannelConfig.from_dict(ch),
        modulations=tuple(d.get("modulations", MODS_DEFAULT)),
        n_symbols=int(d.get("n_symbols", 128)),
        sps=int(d.get("sps", 4)),
        rolloff=float(d.get("rolloff", 0.35)),
        seed=seed,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[float, float]:
    """Return ``(accuracy, cross_entropy_loss)``. **loss is the primary metric for the sensitivity analysis.**"""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    for i in range(0, len(yt), batch_size):
        xb = xt[i : i + batch_size].to(device)
        yb = yt[i : i + batch_size].to(device)
        logits = model(xb)
        total_loss += float(crit(logits, yb).item())
        correct += int((logits.argmax(1) == yb).sum().item())
        n += int(yb.numel())
    return correct / max(n, 1), total_loss / max(n, 1)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
