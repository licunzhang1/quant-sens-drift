"""torch 侧的逐层选择性 fake-quant。

语义必须与 ``src/quant/core.py`` 的 numpy 版本**完全一致**，否则
冒烟测试（numpy）与真实实验（torch）不可比。

为什么不直接用 ``torch.quantization``
------------------------------------
本项目需要**精确控制只量化第 ℓ 层**、per-channel 与 per-tensor 的差别，
以及"仅权重 / 权重+激活"的组合。框架的抽象在这些点上会挡路，
而且不同 torch 版本的行为不一致。手写 fake-quant 只有几十行，完全可控。

参考：Jacob et al., *Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference* (2018) —— 本文采用同样的
仿射量化形式与 STE（直通估计）梯度。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "fake_quant_tensor",
    "quantizable_layers",
    "quantize_model",
    "quantize_one_layer",
    "count_parameters",
]

_EPS = 1e-12
_QUANTIZABLE = (nn.Linear, nn.Conv1d, nn.Conv2d)


# --------------------------------------------------------------------------
# STE
# --------------------------------------------------------------------------
class _STEQuantize(torch.autograd.Function):
    """``round`` 的梯度处处为 0，直接用会让反向传播断掉。
    这里用直通估计（STE）：反向时把梯度原样传过去。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
                qmin: int, qmax: int) -> torch.Tensor:  # noqa: D102
        q = torch.clamp(torch.round(x / scale + zero), qmin, qmax)
        return (q - zero) * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D102
        return grad_output, None, None, None, None


# --------------------------------------------------------------------------
# 量化参数
# --------------------------------------------------------------------------
def _reduce_dims(ndim: int, axis: Optional[int]) -> Optional[List[int]]:
    if axis is None:
        return None
    ax = axis % ndim
    return [i for i in range(ndim) if i != ax]


def _qrange(bits: int, symmetric: bool) -> Tuple[int, int]:
    if bits < 1:
        raise ValueError(f"bits 必须 >= 1，收到 {bits}")
    if symmetric:
        if bits < 2:
            raise ValueError("对称量化在 bits=1 时无意义，请用 asymmetric")
        qmax = 2 ** (bits - 1) - 1
        return -qmax, qmax
    return 0, 2 ** bits - 1


def fake_quant_tensor(
    x: torch.Tensor,
    bits: int,
    symmetric: bool = True,
    per_channel_axis: Optional[int] = None,
    eps: float = _EPS,
    use_ste: bool = True,
) -> torch.Tensor:
    """量化后立刻反量化（fake quantization）。形状与 ``x`` 相同。

    Parameters
    ----------
    per_channel_axis : None 表示 per-tensor；给整数表示沿该轴逐通道求 scale。
        权重通常用 axis=0（输出通道），激活用 None。
    """
    if not torch.is_floating_point(x):
        x = x.float()
    qmin, qmax = _qrange(bits, symmetric)
    dims = _reduce_dims(x.dim(), per_channel_axis)

    if symmetric:
        amax = x.abs().amax(dim=dims, keepdim=True) if dims is not None else x.abs().amax()
        scale = torch.clamp(amax, min=eps) / qmax
        zero = torch.zeros_like(scale)
    else:
        if dims is None:
            xmin, xmax = x.min(), x.max()
        else:
            xmin = x.amin(dim=dims, keepdim=True)
            xmax = x.amax(dim=dims, keepdim=True)
        scale = torch.clamp(xmax - xmin, min=eps) / qmax
        zero = torch.round(-xmin / scale).clamp(qmin, qmax)

    if use_ste:
        return _STEQuantize.apply(x, scale, zero, qmin, qmax)
    q = torch.clamp(torch.round(x / scale + zero), qmin, qmax)
    return (q - zero) * scale


# --------------------------------------------------------------------------
# 层发现
# --------------------------------------------------------------------------
def quantizable_layers(model: nn.Module) -> List[str]:
    """返回可量化的层名（按网络定义顺序）。"""
    return [n for n, m in model.named_modules() if isinstance(m, _QUANTIZABLE)]


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------
# 量化作用域
# --------------------------------------------------------------------------
def _make_act_hook(bits: int, symmetric: bool):
    def hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            return fake_quant_tensor(output, bits, symmetric=symmetric,
                                     per_channel_axis=None)
        return output

    return hook


@contextmanager
def quantize_model(
    model: nn.Module,
    bits: int,
    *,
    weight: bool = True,
    act: bool = False,
    per_channel: bool = True,
    symmetric: bool = True,
    only: Optional[Iterable[str]] = None,
    skip: Sequence[str] = (),
) -> Iterator[None]:
    """临时把模型的权重和/或激活量化到 ``bits`` 位，退出时完整恢复。

    Parameters
    ----------
    only : 只量化这些层（None = 全部可量化层）。**逐层敏感性分析用这个。**
    skip : 要排除的层名前缀。
    act  : 是否同时量化激活（用前向 hook）。注意 per-channel 只对权重有意义。

    Notes
    -----
    恢复时用 ``clone()`` 保存的原始权重写回，并且**在 finally 里执行**，
    保证异常时模型不会被留在被污染的状态。
    """
    targets = set(only) if only is not None else set(quantizable_layers(model))
    if skip:
        targets = {t for t in targets if not any(t.startswith(s) for s in skip)}
    if not targets and not act:
        # 没有可量化层时静默通过（例如整层被 skip）
        yield
        return

    saved_w: Dict[str, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    try:
        with torch.no_grad():
            for name, module in model.named_modules():
                if name not in targets:
                    continue
                if weight and getattr(module, "weight", None) is not None:
                    saved_w[name] = module.weight.detach().clone()
                    axis = 0 if (per_channel and module.weight.dim() > 1) else None
                    module.weight.copy_(
                        fake_quant_tensor(module.weight.detach(), bits,
                                          symmetric=symmetric,
                                          per_channel_axis=axis,
                                          use_ste=False)
                    )
                if act:
                    hooks.append(module.register_forward_hook(
                        _make_act_hook(bits, symmetric)))
        yield
    finally:
        for h in hooks:
            h.remove()
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in saved_w:
                    module.weight.copy_(saved_w[name])


@contextmanager
def quantize_one_layer(
    model: nn.Module,
    layer_name: str,
    bits: int,
    *,
    per_channel: bool = True,
    symmetric: bool = True,
    act: bool = False,
) -> Iterator[None]:
    """只量化一层。逐层敏感性矩阵的主循环用它。"""
    with quantize_model(model, bits, weight=True, act=act,
                        per_channel=per_channel, symmetric=symmetric,
                        only=[layer_name]):
        yield
