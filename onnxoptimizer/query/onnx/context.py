import onnxruntime as ort

from onnxoptimizer.query.onnx.model import ModelObject
from onnxoptimizer.query.types.mapper import numpy_type_map


class ModelContext:
    def __init__(self, model_obj: ModelObject):
        self.model_obj = model_obj

        model_data = self.model_obj.model.SerializeToString()
        self.infer_session = ort.InferenceSession(model_data)

        self.labels_map = {
            elem.name: elem for elem in self.infer_session.get_outputs()
            if elem.name.endswith("label") or elem.name.endswith("variable")
        }
        self.probabilities_map = {
            elem.name: elem for elem in self.infer_session.get_outputs()
            if elem.name.endswith("probability")
        }

        self.infer_input = {}

    def return_type(self, which=None):
        if which is None:
            assert len(self.labels_map) == 1
            return next(iter(self.labels_map.values())).type
        else:
            return self.labels_map[which].type

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

        label_out = []
        labels = list(self.labels_map.keys())
        for elem in labels:
            label_out.append(elem.replace("output_label", "").replace("variable", ""))

        infer_res = session.run(labels, infer_batch)
        infer_result = {}

        for i in range(len(labels)):
            infer_result[label_out[i]] = infer_res[i]

        if len(infer_result) > 1:
            ret = infer_result
        else:
            ret = infer_res[0]

        return ret
