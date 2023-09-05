from onnxoptimizer.query.pandas.computation.eval import eval
import gorilla
import pandas


@gorilla.patch(pandas)
def my_eval(expr: str):
    print("This is my expr eval patching into pandas.")
    return eval(expr)