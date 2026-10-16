"""模型集合。

``NumpyMLP`` 是纯 numpy 的参考实现，急切导入。
``CLDNN`` / ``ResNet1D`` 依赖 torch，**惰性导入** —— 这样只跑
``tests/test_quant_core.py`` 与 ``tests/smoke_pipeline.py`` 的机器
（只有 numpy）不会被拖累。
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
