"""纯 numpy 的参考 MLP —— 用于**独立验证**整条实验管线。

存在的意义
----------
项目的科学结论必须能被独立复核。这个模块不含任何深度学习框架依赖，
用最朴素的写法实现"逐层选择性量化 → 前向 → 测指标"这条链路。

用途：
  1. ``tests/smoke_pipeline.py`` 用它跑通端到端流程（几秒内完成）；
  2. 作为 torch 版本的**语义参照** —— 两边对同一份数据应给出同量级的结果；
  3. 答辩时可以说："核心管线有一个不依赖框架的参考实现。"

它不是用来刷精度的，结构刻意保持最小（MLP + ReLU + softmax）。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..quant.core import quantize_dequantize

__all__ = ["NumpyMLP", "quantized_layer"]


class NumpyMLP:
    """最小 MLP：``dims = [in_dim, h1, ..., n_classes]``。"""

    def __init__(self, dims: Sequence[int], seed: int = 0):
        if len(dims) < 2:
            raise ValueError("dims 至少要有 2 项")
        self.dims = [int(d) for d in dims]
        self.n_layers = len(self.dims) - 1
        rng = np.random.default_rng(seed)

        self.params: Dict[str, np.ndarray] = {}
        for i in range(self.n_layers):
            fan_in = self.dims[i]
            # He 初始化
            W = rng.standard_normal((self.dims[i + 1], fan_in)) * np.sqrt(2.0 / fan_in)
            b = np.zeros(self.dims[i + 1], dtype=np.float64)
            self.params[f"fc{i}.weight"] = W.astype(np.float64)
            self.params[f"fc{i}.bias"] = b

    # ------------------------------------------------------------------ 结构
    def weight_layer_names(self) -> List[str]:
        """可量化的权重层名（供敏感性分析遍历）。"""
        return [f"fc{i}.weight" for i in range(self.n_layers)]

    def layer_costs(self) -> np.ndarray:
        """每层的参数量，作为比特预算的权重。"""
        return np.array(
            [self.params[n].size for n in self.weight_layer_names()], dtype=np.float64
        )

    # ------------------------------------------------------------------ 前向
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

    # ------------------------------------------------------------------ 训练
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
        """Adam + softmax 交叉熵。返回每轮的平均损失。"""
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
                # 反传到上一层的 ReLU 之前：掩码由上一层的预激活决定
                dh = d @ self.params[f"fc{i}.weight"]
                d = dh * (cache[i - 1][1] > 0)
        return grads


# --------------------------------------------------------------------------
# 选择性量化
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
    """临时把 ``layer_name`` 量化到 ``bits`` 位，退出时恢复。

    这是 numpy 版的量化作用域，与 torch 版 ``fake_quant.quantize_model`` 语义一致。
    """
    if layer_name not in model.params:
        raise KeyError(f"模型中没有参数 {layer_name!r}")
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
    """把所有**权重**（不含 bias）量化到 ``bits`` 位。用于全模型量化基线。"""
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
# 数值工具
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
