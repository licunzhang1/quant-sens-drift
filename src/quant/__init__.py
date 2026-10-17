from .core import (
    DEFAULT_BITS_LADDER,
    channel_wise_nmse,
    dequantize,
    error_curves_to_matrix,
    qrange,
    quant_error_curve,
    quant_params,
    quantize,
    quantize_dequantize,
    reconstruction_error,
)
from .schemes import QuantScheme, default_scheme_grid, scheme_from_name

__all__ = [
    "DEFAULT_BITS_LADDER",
    "channel_wise_nmse",
    "dequantize",
    "error_curves_to_matrix",
    "qrange",
    "quant_error_curve",
    "quant_params",
    "quantize",
    "quantize_dequantize",
    "reconstruction_error",
    "QuantScheme",
    "default_scheme_grid",
    "scheme_from_name",
    "fake_quant_tensor",
    "quantizable_layers",
    "quantize_model",
    "quantize_one_layer",
]


def __getattr__(name: str):
    """Import the torch-side helpers on demand.

    That way ``import src.quant`` also works on a machine without torch (for
    example one that only runs ``tests/test_quant_core.py``).
    """
    if name in {"fake_quant_tensor", "quantizable_layers",
                "quantize_model", "quantize_one_layer"}:
        from . import fake_quant as _fq

        return getattr(_fq, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
