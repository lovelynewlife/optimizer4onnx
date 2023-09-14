import gorilla
import onnxoptimizer.query.pandas.api.patch as pdp
from onnxoptimizer.query.pandas.api.function import model_udf

patches = gorilla.find_patches([pdp])

for patch in patches:
    gorilla.apply(patch)

__all__ = [
    "model_udf"
]
