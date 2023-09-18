import onnxoptimizer
from onnxoptimizer.joint import merge_project_models
from onnxoptimizer.query.onnx.context import ModelContext
from onnxoptimizer.query.onnx.model import ModelObject


class MultiModelExprOptimizer:
    def __init__(self):
        self.rules = []
        self.model_optimizer = onnxoptimizer

    def optimize(self, expr_list):
        models = []
        all_inputs = {}
        for expr in expr_list:
            kwargs = expr.terms.inputs
            model = expr.terms.model
            all_inputs.update(kwargs)
            models.append((expr.assigner, model))

        assert len(models) >= 2

        model0_prefix, model0_model = models[0]
        model1_prefix, model1_model = models[1]
        model_fused = merge_project_models(model0_model, model1_model, model0_prefix, model1_prefix)

        for i in range(2, len(models)):
            model_prefix, model_model = models[i]
            model_fused = merge_project_models(model_fused, model_model, "", model_prefix)

        model_fused = self.model_optimizer.optimize(model_fused, fixed_point=True)

        model_obj = ModelObject(model_fused)

        model_context = ModelContext(model_obj)

        model_context.set_infer_input(**all_inputs)

        return model_context
