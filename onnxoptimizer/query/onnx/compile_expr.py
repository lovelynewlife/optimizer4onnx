import numpy as np
import onnx.numpy_helper
from onnx import ValueInfoProto
from onnx.compose import merge_models

from onnxoptimizer.query.onnx.joint.fragment import ModelFragment, OpModelFragment, TermModelFragment
from onnxoptimizer.query.onnx.joint.merge_expr import merge_models_binary_wrap, merge_models_unary_wrap
from onnxoptimizer.query.pandas.core.computation.ops import (ONNXPredicate, ONNXFuncNode,
                                                             BinOp, Term, UnaryOp, Constant)
from onnxoptimizer.query.types.mapper import numpy_onnx_tensor_type_map

ONNX_CMP_OPS_SYMS = (">", "<", ">=", "<=", "==", "!=")
_onnx_cmp_ops_nodes = (
    "Greater",
    "Less",
    "GreaterOrEqual",
    "LessOrEqual",
    "Equal",
    "NotEqual"  # a placeholder cause onnx do not support 'NotEqual' op
)
_onnx_cmp_ops_map = dict(zip(ONNX_CMP_OPS_SYMS, _onnx_cmp_ops_nodes))

ONNX_BOOL_OPS_SYMS = ("&", "|", "and", "or")
_onnx_bool_ops_nodes = (
    "And",
    "Or",
    "Not",
    "And",
    "Or",
    "Not",
)
_onnx_bool_ops_map = dict(zip(ONNX_BOOL_OPS_SYMS, _onnx_bool_ops_nodes))

ONNX_ARITH_OPS_SYMS = ("+", "-", "*", "/", "**", "//", "%")
_onnx_arith_ops_nodes = (
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "DivFloor",  # a placeholder cause onnx do not support 'DivFloor' op
    "Mod"
)
_onnx_arith_ops_map = dict(zip(ONNX_ARITH_OPS_SYMS, _onnx_arith_ops_nodes))

_onnx_binary_ops_dict = {}

for d in (_onnx_bool_ops_map, _onnx_cmp_ops_map, _onnx_arith_ops_map):
    _onnx_binary_ops_dict.update(d)

ONNX_UNARY_OPS_SYMS = ("+", "-", "~", "not")
_onnx_unary_ops_nodes = (
    "Identity",
    "Neg",
    "Not",
    "Not"
)
_onnx_unary_ops_map = dict(zip(ONNX_UNARY_OPS_SYMS, _onnx_unary_ops_nodes))


class ONNXPredicateCompiler:
    helper = onnx.helper
    numpy_helper = onnx.numpy_helper

    IDENTITY = "Identity"

    def __init__(self, env):
        self.env = env
        self.temps = 0

    def compile_save(self, node, path=None):
        model = self.compile(node)
        onnx.save(model.model, path)

    def compile(self, node) -> ModelFragment:
        compile_method = f"compile_{type(node).__name__}"
        compile_func = getattr(self, compile_method)

        return compile_func(node)

    def get_temp_name_by_value(self, value):
        name = f"{type(value).__name__}_{self.temps}"
        self.temps += 1

        return name

    def get_temp_name(self):
        name = f"temp_{self.temps}"
        self.temps += 1

        return name

    def compile_ONNXPredicate(self, node: ONNXPredicate):
        return self.compile_BinOp(node)

    def compile_ONNXFuncNode(self, node: ONNXFuncNode) -> ModelFragment:
        return ModelFragment(node.model, node.return_type, node.model_context.infer_input)

    def compile_BinOp(self, node: BinOp):
        lhs = node.lhs
        rhs = node.rhs
        op = node.op

        lhs_ir = self.compile(lhs)
        rhs_ir = self.compile(rhs)

        lhs_prefix = self.get_temp_name()
        rhs_prefix = self.get_temp_name()

        lhs_tensor = lhs_ir.get_default_output()
        lhs_name = lhs_prefix + lhs_tensor.name

        rhs_tensor = rhs_ir.get_default_output()
        rhs_name = rhs_prefix + rhs_tensor.name

        lhs_input_tensor = ValueInfoProto()
        lhs_input_tensor.CopyFrom(lhs_tensor)
        lhs_input_tensor.name = lhs_name

        rhs_input_tensor = ValueInfoProto()
        rhs_input_tensor.CopyFrom(rhs_tensor)
        rhs_input_tensor.name = rhs_name

        output_type = numpy_onnx_tensor_type_map[node.return_type.type]

        output_name = self.get_temp_name()

        # TODO: How to determine proper shape?
        output_tensor = self.helper.make_tensor_value_info(
            name=output_name,
            elem_type=output_type,
            shape=[None]
        )

        op_node = self.helper.make_node(
            op_type=_onnx_binary_ops_dict[op],
            inputs=[lhs_name, rhs_name],
            outputs=[output_name]
        )

        partial_graph = self.helper.make_graph(
            nodes=[op_node],
            name=self.get_temp_name(),
            inputs=[lhs_input_tensor, rhs_input_tensor],
            outputs=[output_tensor]
        )

        partial_model = self.helper.make_model(partial_graph)

        partial_model = merge_models_binary_wrap(
            lhs_ir.model,
            rhs_ir.model,
            partial_model,
            prefix1=lhs_prefix,
            prefix2=rhs_prefix
        )

        external_input = {}
        if lhs_ir.external_input is not None:
            external_input.update(lhs_ir.external_input)
        if rhs_ir.external_input is not None:
            external_input.update(rhs_ir.external_input)

        partial_frag = OpModelFragment(partial_model, node.return_type, op, external_input)

        return partial_frag

    def compile_UnaryOp(self, node: UnaryOp):
        op = node.op
        operand = self.compile(node.operand)

        output_name = self.get_temp_name()

        input_prefix = self.get_temp_name()
        input_tensor = operand.get_default_output()
        input_name = input_prefix + input_tensor.name

        output_type = numpy_onnx_tensor_type_map[node.return_type.type]

        # Should do shape inference here?
        output_shape = input_tensor.type.tensor_type.shape

        output_tensor = self.helper.make_tensor_value_info(
            name=output_name,
            elem_type=output_type,
            shape=output_shape
        )

        op_node = self.helper.make_node(
            op_type=_onnx_unary_ops_map[op],
            inputs=[input_name],
            outputs=[output_name]
        )

        partial_graph = self.helper.make_graph(
            nodes=[op_node],
            name=self.get_temp_name(),
            inputs=[input_tensor],
            outputs=[output_tensor]
        )

        partial_model = self.helper.make_model(partial_graph)

        merge_model = merge_models_unary_wrap(
            m1=operand.model,
            mop=partial_model,
            prefix1=input_prefix,
        )

        merge_frag = OpModelFragment(
            merge_model, operand.return_type, op, external_input=operand.external_input
        )

        return merge_frag

    def compile_as_initializer(self, node):
        output_name = self.get_temp_name()
        value_name = self.get_temp_name_by_value(node.value)

        np_value = np.array(node.value)
        constant_value = self.numpy_helper.from_array(np_value, name=value_name)

        identity_node = self.helper.make_node(self.IDENTITY,
                                              inputs=[value_name],
                                              outputs=[output_name])

        term_output = self.helper.make_tensor_value_info(name=output_name,
                                                         elem_type=constant_value.data_type,
                                                         shape=np_value.shape)

        partial_graph = self.helper.make_graph(
            nodes=[identity_node],
            name=self.get_temp_name(),
            inputs=[],
            outputs=[term_output],
            initializer=[constant_value]
        )

        partial_model = self.helper.make_model(partial_graph)

        return TermModelFragment(partial_model, np_value.dtype)

    def compile_Term(self, node: Term):
        return self.compile_as_initializer(node)

    def compile_Constant(self, node: Constant):
        return self.compile_as_initializer(node)
