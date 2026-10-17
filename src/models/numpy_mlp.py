"""Pure-numpy reference MLP -- used to **independently validate** the pipeline.

Why it exists
-------------
The scientific conclusions of the project have to be independently checkable.
This module pulls in no deep-learning framework; it implements the chain
"layer-wise selective quantization -> forward pass -> metric" in the most
straightforward way possible.

Uses:
  1. ``tests/smoke_pipeline.py`` runs the end-to-end flow with it (finishes in
     a few seconds);
  2. it is the **semantic reference** for the torch version -- both should give
     results of the same order on the same data;
  3. in a defense one can say: "the core pipeline has a reference
     implementation that depends on no framework."

It is not meant for chasing accuracy; the structure is deliberately minimal
(MLP + ReLU + softmax).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..quant.core import quantize_dequantize

__all__ = ["NumpyMLP", "quantized_layer"]


class NumpyMLP:
    """Minimal MLP: ``dims = [in_dim, h1, ..., n_classes]``."""

    def __init__(self, dims: Sequence[int], seed: int = 0):
        if len(dims) < 2:
            raise ValueError("dims needs at least 2 entries")
        self.dims = [int(d) for d in dims]
        self.n_layers = len(self.dims) - 1
        rng = np.random.default_rng(seed)

        self.params: Dict[str, np.ndarray] = {}
        for i in range(self.n_layers):
            fan_in = self.dims[i]
            # He initialization
            W = rng.standard_normal((self.dims[i + 1], fan_in)) * np.sqrt(2.0 / fan_in)
            b = np.zeros(self.dims[i + 1], dtype=np.float64)
            self.params[f"fc{i}.weight"] = W.astype(np.float64)
            self.params[f"fc{i}.bias"] = b

    # ------------------------------------------------------------ structure
    def weight_layer_names(self) -> List[str]:
        """Names of the quantizable weight layers (for sensitivity sweeps)."""
        return [f"fc{i}.weight" for i in range(self.n_layers)]

    def layer_costs(self) -> np.ndarray:
        """Parameter count per layer, used as the weight of the bit budget."""
        return np.array(
            [self.params[n].size for n in self.weight_layer_names()], dtype=np.float64
        )

    # -------------------------------------------------------------- forward
    def forward(self, x: np.ndarray, return_cache: bool = False):
        h = np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)
        cache = []
        for i in range(self.n_layers):
            W = self.params[f"fc{i}.weight"]
            b = self.params[f"fc{i}.bias"]
            z = h @ W.T + b
            cache.append((h, z))
            if i < self.n_layers - 1:
                h = np.maximum(z, 0.0)
            else:
                h = z
        if return_cache:
            return h, cache
        return h

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        logits = self.forward(x)
        return float(_softmax_ce(logits, y))

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        logits = self.forward(x)
        return float(np.mean(np.argmax(logits, axis=1) == y))

    # ------------------------------------------------------------- training
    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        verbose: bool = False,
    ) -> List[float]:
        """Adam with softmax cross-entropy. Returns the mean loss per epoch."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n = x.shape[0]
        rng = np.random.default_rng(seed)

        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        history: List[float] = []

        for ep in range(epochs):
            perm = rng.permutation(n)
            total = 0.0
            nb = 0
            for s in range(0, n, batch_size):
                idx = perm[s : s + batch_size]
                xb, yb = x[idx], y[idx]
                grads = self._backward(xb, yb)
                step += 1
                for k in self.params:
                    g = grads[k] + weight_decay * self.params[k]
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * g * g
                    mh = m[k] / (1 - b1 ** step)
                    vh = v[k] / (1 - b2 ** step)
                    self.params[k] -= lr * mh / (np.sqrt(vh) + eps)
                total += self.loss(xb, yb)
                nb += 1
            epoch_loss = total / max(nb, 1)
            history.append(epoch_loss)
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"    epoch {ep:>3}  loss={epoch_loss:.4f}  acc={self.accuracy(x, y):.4f}")
        return history

    def _backward(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        logits, cache = self.forward(x, return_cache=True)
        n = x.shape[0]
        probs = _softmax(logits)
        d = probs.copy()
        d[np.arange(n), y] -= 1.0
        d /= n

        grads: Dict[str, np.ndarray] = {}
        for i in range(self.n_layers - 1, -1, -1):
            h_in, _z = cache[i]
            grads[f"fc{i}.weight"] = d.T @ h_in
            grads[f"fc{i}.bias"] = d.sum(axis=0)
            if i > 0:
                # backprop to before the previous layer's ReLU: the mask
                # comes from that layer's pre-activation
                dh = d @ self.params[f"fc{i}.weight"]
                d = dh * (cache[i - 1][1] > 0)
        return grads


# --------------------------------------------------------------------------
# selective quantization
# --------------------------------------------------------------------------
@contextmanager
def quantized_layer(
    model: NumpyMLP,
    layer_name: str,
    bits: int,
    *,
    per_channel: bool = True,
    symmetric: bool = True,
) -> Iterator[None]:
    """Temporarily quantize ``layer_name`` to ``bits`` bits, restoring on exit.

    This is the numpy quantization scope; it matches the semantics of the torch
    ``fake_quant.quantize_model``.
    """
    if layer_name not in model.params:
        raise KeyError(f"no parameter {layer_name!r} in the model")
    orig = model.params[layer_name]
    axis = 0 if (per_channel and orig.ndim > 1) else None
    xq, _, _ = quantize_dequantize(orig, bits, symmetric, axis)
    model.params[layer_name] = xq
    try:
        yield
    finally:
        model.params[layer_name] = orig


@contextmanager
def quantized_all_weights(
    model: NumpyMLP,
    bits: int,
    *,
    per_channel: bool = True,
    symmetric: bool = True,
) -> Iterator[None]:
    """Quantize all **weights** (bias excluded) to ``bits`` bits.

    Used for the full-model quantization baseline.
    """
    saved = {}
    for name in model.weight_layer_names():
        orig = model.params[name]
        axis = 0 if (per_channel and orig.ndim > 1) else None
        saved[name] = orig
        model.params[name], _, _ = quantize_dequantize(orig, bits, symmetric, axis)
    try:
        yield
    finally:
        for name, orig in saved.items():
            model.params[name] = orig


# --------------------------------------------------------------------------
# numerical helpers
# --------------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _softmax_ce(logits: np.ndarray, y: np.ndarray) -> float:
    p = _softmax(logits)
    n = y.shape[0]
    p_true = np.clip(p[np.arange(n), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p_true)))
