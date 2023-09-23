import onnx

import onnxoptimizer
from onnxoptimizer.query.onnx.compile_expr import ONNXPredicateCompiler
from onnxoptimizer.query.onnx.joint import merge_project_models_wrap
from onnxoptimizer.query.onnx.context import ModelContext
from onnxoptimizer.query.onnx.model import ModelObject
from onnxoptimizer.query.pandas.core.computation.expr import ComposedExpr
from onnxoptimizer.query.pandas.core.computation.ops import ONNXFuncNode, ONNXPredicate


class ModelExprOptimizer:
    def __init__(self, env, engine, level):
        self.model_optimizer = onnxoptimizer
        self.env = env
        self.engine = engine
        self.level = level
        self.predicate_compiler = ONNXPredicateCompiler(env)

    def optimize(self, expr_list):
        expr_opted = []

        expr_to_opt = []
        assigners = []

        predicate_to_opt = []

        for e2e in expr_list:
            if isinstance(e2e.terms, ONNXFuncNode) and e2e.assigner is not None:
                expr_to_opt.append(e2e)
                assigners.append(e2e.assigner)
            if isinstance(e2e.terms, ONNXPredicate):
                predicate_to_opt.append(e2e)
            else:
                expr_opted.append(e2e)

        if len(expr_to_opt) < 2:
            expr_opted.extend(expr_to_opt)
        else:
            fused_expr = self._optimize_multi_expr(expr_to_opt, assigners)
            expr_opted.extend(fused_expr)

        opted_predicates = self._optimize_predicate(predicate_to_opt)

        expr_opted.extend(opted_predicates)

        return expr_opted

    def _optimize_multi_expr(self, expr_list, assigners):
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
        model_fused = merge_project_models_wrap(model0_model, model1_model, model0_prefix, model1_prefix)

        for i in range(2, len(models)):
            model_prefix, model_model = models[i]
            model_fused = merge_project_models_wrap(model_fused, model_model, None, model_prefix)

        model_fused = self.model_optimizer.optimize(model_fused, fixed_point=True)

        model_obj = ModelObject(model_fused)

        model_context = ModelContext(model_obj)

        model_context.set_infer_input(**all_inputs)

        composed_expr = ComposedExpr(self.engine, self.env, self.level,
                                     model_context, assigners)

        return [composed_expr]

    def _optimize_predicate(self, expr_list):
        opted_expr_list = []

        for expr in expr_list:
            compiled = self.predicate_compiler.compile(expr.terms)

            model_obj = ModelObject(compiled.model_partial)
            compiled_term = ModelContext(model_obj)
            if compiled.external_input is not None:
                compiled_term.set_infer_input(**compiled.external_input)
            else:
                compiled_term.set_infer_input(**{})
            compiled_expr = ComposedExpr(self.engine, self.env, self.level, compiled_term)

            opted_expr_list.append(compiled_expr)

        return opted_expr_list



