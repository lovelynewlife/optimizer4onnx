from typing import Optional

import numpy as np
from onnx import ModelProto

from onnxoptimizer.query.types.mapper import numpy_onnx_tensor_type_map


class ModelFragment:
    def __init__(self, model_partial: ModelProto, return_type,
                 external_input: Optional[dict] = None):
        self.external_input = external_input
        self.model_partial = model_partial
        self.return_type = return_type

    @property
    def model(self):
        return self.model_partial

    @property
    def model_inputs(self):
        return list(self.model_partial.graph.input)

    @property
    def model_outputs(self):
        return list(self.model_partial.graph.output)

    def return_tensor_type(self):
        return numpy_onnx_tensor_type_map[self.return_type]

    def get_outputs_endswith(self, suffix: str):
        outputs = []

        for elem in self.model_outputs:
            if elem.name.endswith(suffix):
                outputs.append(elem)

        return outputs

    def get_default_output(self):
        maybe_output = self.get_outputs_endswith("_label")
        maybe_output.extend(self.get_outputs_endswith("_variable"))
        return maybe_output[0]

    def get_default_input(self):
        return self.model_inputs[0]


class OpModelFragment(ModelFragment):
    def __init__(self, model_partial: ModelProto, return_type,
                 op: str, external_input: Optional[dict] = None):
        super().__init__(model_partial, return_type, external_input)
        self.op = op

    def get_default_output(self):
        return self.model_outputs[0]

    def get_default_input(self):
        return self.model_inputs[0]


class TermModelFragment(ModelFragment):
    def get_default_output(self):
        return self.model_outputs[0]

    def get_default_input(self):
        return self.model_inputs[0]
