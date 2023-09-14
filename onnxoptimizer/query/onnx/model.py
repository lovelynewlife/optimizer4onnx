import onnx
import skl2onnx
from sklearn.pipeline import Pipeline

import onnxoptimizer
from onnxoptimizer.query.types.mapper import numpy_onnx_type_map


class ModelObject:
    def __init__(self, pipeline: str | Pipeline, schema=None):
        self.pipeline = pipeline
        self.schema = schema

        if type(pipeline) == str:
            self.model = onnx.load_model(self.pipeline)
        else:
            init_types = [(k, numpy_onnx_type_map[v]) for k, v in self.schema.items()]
            self.model = skl2onnx.to_onnx(self.pipeline, initial_types=init_types)

        self.model = onnxoptimizer.optimize(self.model)
