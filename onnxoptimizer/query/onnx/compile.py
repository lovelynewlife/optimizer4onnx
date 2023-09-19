from onnx.compose import merge_graphs, merge_models

from onnxoptimizer.query.pandas.core.computation.ops import ONNXPredicate, ONNXFuncNode, BinOp, Term, UnaryOp

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
    def __init__(self, env):
        self.env = env

    def compile(self, node):
        compile_method = f"compile_{type(node).__name__}"
        compile_func = getattr(self, compile_method)

        return compile_func(node)

    def compile_ONNXPredicate(self, node: ONNXPredicate):
        lhs = node.lhs
        rhs = node.rhs
        op = node.op

        lhs_ir = self.compile(lhs)
        rhs_ir = self.compile(rhs)
        print("compile predicate:", op)

    def compile_ONNXFuncNode(self, node: ONNXFuncNode):
        print("compile func node:", node.name)

    def compile_BinOp(self, node: BinOp):
        lhs = node.lhs
        rhs = node.rhs
        op = node.op

        lhs_ir = self.compile(lhs)
        rhs_ir = self.compile(rhs)
        print("compile normal bin op:", op)

    def compile_UnaryOp(self, node: UnaryOp):
        op = node.op
        operand = self.compile(node.operand)
        print("compile normal unary op:", op)

    def compile_Term(self, node: Term):
        print("compile term node:", node.name)

    def compile_Constant(self, node: Term):
        print("compile constant node:", node.name)
