"""Layer-wise selective fake-quant on the torch side.

The semantics have to match the numpy version in ``src/quant/core.py``
**exactly**, otherwise the smoke tests (numpy) and the real experiments (torch)
are not comparable.

Why not use ``torch.quantization`` directly
-------------------------------------------
This project needs **precise control over quantizing only layer l**, over the
per-channel versus per-tensor difference, and over the "weights only / weights +
activations" combinations. The framework abstractions get in the way at exactly
these points, and behaviour differs between torch versions. A hand-written
fake-quant is a few dozen lines and fully under control.

Reference: Jacob et al., *Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference* (2018); this work uses the same
affine quantization form and the same STE (straight-through estimator) gradient.
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
    """The gradient of ``round`` is zero everywhere, so using it directly breaks
    backpropagation. The straight-through estimator (STE) is used here instead:
    the gradient is passed through unchanged on the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
                qmin: int, qmax: int) -> torch.Tensor:  # noqa: D102
        q = torch.clamp(torch.round(x / scale + zero), qmin, qmax)
        return (q - zero) * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D102
        return grad_output, None, None, None, None


# --------------------------------------------------------------------------
# quantization parameters
# --------------------------------------------------------------------------
def _reduce_dims(ndim: int, axis: Optional[int]) -> Optional[List[int]]:
    if axis is None:
        return None
    ax = axis % ndim
    return [i for i in range(ndim) if i != ax]


def _qrange(bits: int, symmetric: bool) -> Tuple[int, int]:
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    if symmetric:
        if bits < 2:
            raise ValueError("symmetric quantization is meaningless at bits=1; use asymmetric")
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
    """Quantize then immediately dequantize (fake quantization). Shape as ``x``.

    Parameters
    ----------
    per_channel_axis : None means per-tensor; an integer means the scale is
        computed per channel along that axis. Weights normally use axis=0 (the
        output-channel axis), activations use None.
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
# layer discovery
# --------------------------------------------------------------------------
def quantizable_layers(model: nn.Module) -> List[str]:
    """Return the quantizable layer names (in network definition order)."""
    return [n for n, m in model.named_modules() if isinstance(m, _QUANTIZABLE)]


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------
# quantization scopes
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
    """Temporarily quantize the model's weights and/or activations to ``bits``
    bits, restoring everything fully on exit.

    Parameters
    ----------
    only : quantize only these layers (None = every quantizable layer). **This is
        what the layer-wise sensitivity analysis uses.**
    skip : layer-name prefixes to exclude.
    act  : whether to quantize activations as well (via forward hooks). Note that
        per-channel only makes sense for weights.

    Notes
    -----
    Restoration writes back the original weights saved with ``clone()``, and it
    **runs inside a finally block**, so an exception cannot leave the model in a
    corrupted state.
    """
    targets = set(only) if only is not None else set(quantizable_layers(model))
    if skip:
        targets = {t for t in targets if not any(t.startswith(s) for s in skip)}
    if not targets and not act:
        # pass through silently when there is nothing to quantize (e.g. all skipped)
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
    """Quantize a single layer. Used by the main loop of the layer-wise
    sensitivity matrix."""
    with quantize_model(model, bits, weight=True, act=act,
                        per_channel=per_channel, symmetric=symmetric,
                        only=[layer_name]):
        yield
