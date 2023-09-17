import onnx
import skl2onnx
from onnx import ModelProto
from sklearn.pipeline import Pipeline

import onnxoptimizer
from onnxoptimizer.query.types.mapper import numpy_onnx_type_map


class ModelObject:
    def __init__(self, pipeline: str | Pipeline | ModelProto, schema=None):
        self.schema = schema

        if type(pipeline) == str:
            self.model = onnx.load_model(pipeline)
        elif type(pipeline) == Pipeline:
            init_types = [(k, numpy_onnx_type_map[v]) for k, v in self.schema.items()]
            self.model = skl2onnx.to_onnx(pipeline, initial_types=init_types)
        else:
            self.model = pipeline

        self.model = onnxoptimizer.optimize(self.model, fixed_point=True)
