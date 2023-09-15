from __future__ import annotations

from pandas.compat._optional import import_optional_dependency

ne = import_optional_dependency("numexpr", errors="warn")
NUMEXPR_INSTALLED = ne is not None
if NUMEXPR_INSTALLED:
    NUMEXPR_VERSION = ne.__version__
else:
    NUMEXPR_VERSION = None

ort = import_optional_dependency("onnxruntime", errors="warn")
ONNXRUNTIME_INSTALLED = ort is not None
if ONNXRUNTIME_INSTALLED:
    ONNXRUNTIME_VERSION = ort.__version__
else:
    ONNXRUNTIME_VERSION = None

__all__ = ["NUMEXPR_INSTALLED", "NUMEXPR_VERSION"]
