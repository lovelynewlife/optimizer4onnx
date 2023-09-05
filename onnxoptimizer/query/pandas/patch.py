import pandas

from onnxoptimizer.query.pandas.core.computation.eval import pandas_eval
from onnxoptimizer.query.patching import callable_patch

LEVEL_OFFSET_1 = 1


@callable_patch(pandas)
def my_eval(expr: str):
    print("This is my expr eval patching into pandas.")
    return pandas_eval(expr, level=LEVEL_OFFSET_1)
