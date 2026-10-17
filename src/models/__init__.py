"""Model collection.

``NumpyMLP`` is a pure-numpy reference implementation and is imported eagerly.
``CLDNN`` / ``ResNet1D`` depend on torch and are **imported lazily** -- that way
a machine that only runs ``tests/test_quant_core.py`` and
``tests/smoke_pipeline.py`` (numpy only) is not held back.
"""

from .numpy_mlp import NumpyMLP, quantized_all_weights, quantized_layer

__all__ = [
    "NumpyMLP",
    "quantized_all_weights",
    "quantized_layer",
    "CLDNN",
    "ResNet1D",
    "BasicBlock1D",
]

_TORCH_EXPORTS = {"CLDNN", "ResNet1D", "BasicBlock1D"}


def __getattr__(name: str):
    if name in _TORCH_EXPORTS:
        from . import cldnn as _c, resnet1d as _r

        return {"CLDNN": _c.CLDNN, "ResNet1D": _r.ResNet1D,
                "BasicBlock1D": _r.BasicBlock1D}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
