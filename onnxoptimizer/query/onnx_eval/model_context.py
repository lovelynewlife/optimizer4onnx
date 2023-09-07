from typing import List

import numpy as np
import onnx
import onnxruntime as ort

import onnxoptimizer
from onnxoptimizer.joint import merge_project_models

type_map = {
    "int64": np.int64,
    "float64": np.float32,
    "object": str,
}


class ModelContext:
    def __init__(self, path):
        self.path = path

        self.model = onnx.load_model(self.path)
        self.model = onnxoptimizer.optimize(self.model)
        self.model_data = self.model.SerializeToString()

    def __call__(self, **kwargs):
        session = ort.InferenceSession(self.model_data)
        infer_batch = {
            elem: kwargs[elem].to_numpy().astype(type_map[kwargs[elem].dtype.name]).reshape((-1, 1))
            for elem in kwargs.keys()
        }
        labels = [elem.name for elem in session.get_outputs() if elem.name.endswith("output_label")]
        probabilities = [elem.name for elem in session.get_outputs() if elem.name.endswith("output_probability")]
        infer_res = session.run(labels, infer_batch)
        res = {}
        for i in range(len(labels)):
            res[labels[i]] = infer_res[i]
        return res[labels[0]]


class MultiModelContext:
    def __init__(self, exprs):

        self.compose_plan = {}
        models = []
        self.all_inputs = {}
        for expr in exprs:
            kwargs = expr.terms.inputs
            model = expr.terms.model
            self.compose_plan[expr.assigner] = (kwargs, model)
            self.all_inputs.update(kwargs)
            models.append((expr.assigner, model))

        assert len(models) >= 2

        model0_prefix, model0_model = models[0]
        model1_prefix, model1_model = models[1]
        model_fused = merge_project_models(model0_model, model1_model, model0_prefix, model1_prefix)

        for i in range(2, len(models)):
            model_prefix, model_model = models[i]
            model_fused = merge_project_models(model_fused, model_model, "", model_prefix)

        self.model_data = onnxoptimizer.optimize(model_fused).SerializeToString()

    def __call__(self):
        session = ort.InferenceSession(self.model_data)
        infer_batch = {
            elem: self.all_inputs[elem].to_numpy().astype(type_map[self.all_inputs[elem].dtype.name]).reshape((-1, 1))
            for elem in self.all_inputs.keys()
        }
        labels = [elem.name for elem in session.get_outputs() if elem.name.endswith("output_label")]
        probabilities = [elem.name for elem in session.get_outputs() if elem.name.endswith("output_probability")]
        infer_res = session.run(labels, infer_batch)

        label_out = []
        for elem in labels:
            label_out.append(elem.replace("output_label", ""))

        res = {}
        for i in range(len(labels)):
            res[label_out[i]] = infer_res[i]
        return res
