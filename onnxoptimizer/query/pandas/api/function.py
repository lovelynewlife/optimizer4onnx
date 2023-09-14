from typing import Callable, Any

from sklearn.pipeline import Pipeline

from onnxoptimizer.query.onnx.context import ModelContext, ModelObject


class ModelContextAnnotation:
    def __init__(self, model_obj: ModelObject, func: Callable[[ModelContext, ...], Any]):
        self.func = func
        self.model_obj = model_obj

    def get_inputs(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def model(self):
        return self.model_obj.model


def model_udf(path: str | Pipeline, schema=None):
    model_obj = ModelObject(path, schema)

    def model_eval(func: Callable[[...], Any]):
        return ModelContextAnnotation(model_obj, func)

    return model_eval
