import onnxruntime as ort

import onnxoptimizer
from onnxoptimizer.joint import merge_project_models
from onnxoptimizer.query.onnx.model import ModelObject
from onnxoptimizer.query.types.mapper import numpy_type_map


class ModelContext:
    def __init__(self, model_obj: ModelObject):
        self.model_obj = model_obj

        model_data = self.model_obj.model.SerializeToString()
        self.infer_session = ort.InferenceSession(model_data)
        self.infer_input = {}

    def set_infer_input(self, **kwargs):
        self.infer_input = kwargs

    @property
    def model(self):
        return self.model_obj.model

    def __call__(self):
        model_input = self.infer_input
        infer_batch = {
            k: v.to_numpy().astype(numpy_type_map[v.dtype.type]).reshape((-1, 1))
            for k, v in model_input.items()
        }
        session = self.infer_session

        labels = [elem.name for elem in session.get_outputs()
                  if elem.name.endswith("output_label") or elem.name.endswith("variable")]
        probabilities = [elem.name for elem in session.get_outputs() if elem.name.endswith("output_probability")]

        infer_res = session.run(labels, infer_batch)
        res = {}
        for i in range(len(labels)):
            res[labels[i]] = infer_res[i]
        return res[labels[0]]


class MultiModelContext:
    def __init__(self, expr_list):

        self.compose_plan = {}
        models = []
        self.all_inputs = {}
        for expr in expr_list:
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

        self.model_data = onnxoptimizer.optimize(model_fused, fixed_point=True).SerializeToString()

    def __call__(self):
        session = ort.InferenceSession(self.model_data)
        infer_batch = {
            elem: self.all_inputs[elem].to_numpy().astype(numpy_type_map[self.all_inputs[elem].dtype.type]).reshape(
                (-1, 1))
            for elem in self.all_inputs.keys()
        }
        labels = [elem.name for elem in session.get_outputs()
                  if elem.name.endswith("output_label") or elem.name.endswith("variable")]
        probabilities = [elem.name for elem in session.get_outputs() if elem.name.endswith("output_probability")]
        infer_res = session.run(labels, infer_batch)

        label_out = []
        for elem in labels:
            label_out.append(elem.replace("output_label", "").replace("variable", ""))

        res = {}
        for i in range(len(labels)):
            res[label_out[i]] = infer_res[i]
        return res
