"""
Top level ``eval`` module.
"""
from __future__ import annotations

import tokenize
from typing import TYPE_CHECKING
import warnings

from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.common import is_extension_array_dtype

from onnxoptimizer.query.onnx.context import MultiModelContext
from onnxoptimizer.query.pandas.core.computation.engines import ENGINES
from onnxoptimizer.query.pandas.core.computation.expr import (
    PARSERS,
    Expr,
)
from onnxoptimizer.query.pandas.core.computation.parsing import tokenize_string
from onnxoptimizer.query.pandas.core.computation.scope import ensure_scope
from pandas.core.generic import NDFrame

from pandas.io.formats.printing import pprint_thing

from onnxoptimizer.query.pandas.core.computation.ops import BinOp, ONNXEvalNode


def _check_engine(engine: str | None) -> str:
    """
    Make sure a valid engine is passed.

    Parameters
    ----------
    engine : str
        String to validate.

    Raises
    ------
    KeyError
      * If an invalid engine is passed.
    ImportError
      * If numexpr was requested but doesn't exist.

    Returns
    -------
    str
        Engine name.
    """
    from onnxoptimizer.query.pandas.core.computation.check import NUMEXPR_INSTALLED
    from onnxoptimizer.query.pandas.core.computation.expressions import USE_NUMEXPR

    if engine is None:
        engine = "numexpr" if USE_NUMEXPR else "python"

    if engine not in ENGINES:
        valid_engines = list(ENGINES.keys())
        raise KeyError(
            f"Invalid engine '{engine}' passed, valid engines are {valid_engines}"
        )

    # TODO: validate this in a more general way (thinking of future engines
    # that won't necessarily be import-able)
    # Could potentially be done on engine instantiation
    if engine == "numexpr" and not NUMEXPR_INSTALLED:
        raise ImportError(
            "'numexpr' is not installed or an unsupported version. Cannot use "
            "engine='numexpr' for query/eval if 'numexpr' is not installed"
        )

    return engine


def _check_parser(parser: str):
    """
    Make sure a valid parser is passed.

    Parameters
    ----------
    parser : str

    Raises
    ------
    KeyError
      * If an invalid parser is passed
    """
    if parser not in PARSERS:
        raise KeyError(
            f"Invalid parser '{parser}' passed, valid parsers are {PARSERS.keys()}"
        )


def _check_resolvers(resolvers):
    if resolvers is not None:
        for resolver in resolvers:
            if not hasattr(resolver, "__getitem__"):
                name = type(resolver).__name__
                raise TypeError(
                    f"Resolver of type '{name}' does not "
                    "implement the __getitem__ method"
                )


def _check_expression(expr):
    """
    Make sure an expression is not an empty string

    Parameters
    ----------
    expr : object
        An object that can be converted to a string

    Raises
    ------
    ValueError
      * If expr is an empty string
    """
    if not expr:
        raise ValueError("expr cannot be an empty string")


def _convert_expression(expr) -> str:
    """
    Convert an object to an expression.

    This function converts an object to an expression (a unicode string) and
    checks to make sure it isn't empty after conversion. This is used to
    convert operators to their string representation for recursive calls to
    :func:`~pandas.eval`.

    Parameters
    ----------
    expr : object
        The object to be converted to a string.

    Returns
    -------
    str
        The string representation of an object.

    Raises
    ------
    ValueError
      * If the expression is empty.
    """
    s = pprint_thing(expr)
    _check_expression(s)
    return s


def _check_for_locals(expr: str, stack_level: int, parser: str):
    at_top_of_stack = stack_level == 0
    not_pandas_parser = parser != "pandas"

    if not_pandas_parser:
        msg = "The '@' prefix is only supported by the pandas parser"
    elif at_top_of_stack:
        msg = (
            "The '@' prefix is not allowed in top-level eval calls.\n"
            "please refer to your variables by name without the '@' prefix."
        )

    if at_top_of_stack or not_pandas_parser:
        for toknum, tokval in tokenize_string(expr):
            if toknum == tokenize.OP and tokval == "@":
                raise SyntaxError(msg)


def _check_need_fallback(parsed_expr: Expr, engine: str):
    ret_engine = engine

    if ret_engine == "numexpr" and (
            is_extension_array_dtype(parsed_expr.terms.return_type)
            or getattr(parsed_expr.terms, "operand_types", None) is not None
            and any(
        is_extension_array_dtype(elem)
        for elem in parsed_expr.terms.operand_types
    )):
        warnings.warn(
            "Engine has switched to 'python' because numexpr does not support "
            "extension array dtypes. Please set your engine to python manually.",
            RuntimeWarning,
            stacklevel=find_stack_level(),
        )

        ret_engine = "python"

    return ret_engine


def pandas_eval(
        expr: str | BinOp,  # we leave BinOp out of the docstr bc it isn't for users
        parser: str = "pandas",
        engine: str | None = None,
        local_dict=None,
        global_dict=None,
        resolvers=(),
        level: int = 0,
        target=None,
        inplace: bool = False,
        enable_opt: bool = True
):
    inplace = validate_bool_kwarg(inplace, "inplace")

    exprs: list[str | BinOp]
    if isinstance(expr, str):
        _check_expression(expr)
        exprs = [e.strip() for e in expr.splitlines() if e.strip() != ""]
    else:
        # ops.BinOp; for internal compat, not intended to be passed by users
        exprs = [expr]
    multi_line = len(exprs) > 1

    if multi_line and target is None:
        raise ValueError(
            "multi-line expressions are only valid in the "
            "context of data, use DataFrame.eval"
        )
    engine = _check_engine(engine)
    _check_parser(parser)
    _check_resolvers(resolvers)

    ret = None
    first_expr = True
    target_modified = False
    #################
    # Parse Phase
    #################
    expr_to_eval = []
    for expr in exprs:
        expr = _convert_expression(expr)
        _check_for_locals(expr, level, parser)

        # get our (possibly passed-in) scope
        env = ensure_scope(
            level + 1,
            global_dict=global_dict,
            local_dict=local_dict,
            resolvers=resolvers,
            target=target,
        )

        parsed_expr = Expr(expr, engine=engine, parser=parser, env=env)

        expr_to_eval.append(parsed_expr)

    #################
    # Optimization Phase
    #################
    # TODO: optimization phase
    expr_remain = []
    res = {}
    if enable_opt:
        expr_to_opt = []

        for e2e in expr_to_eval:
            if isinstance(e2e.terms, ONNXEvalNode):
                expr_to_opt.append(e2e)
            else:
                expr_remain.append(e2e)

        if len(expr_to_opt) < 2:
            expr_remain.extend(expr_to_opt)
        else:
            # do optimize eval
            fused_expr = MultiModelContext(expr_to_opt)
            res.update(fused_expr())
    else:
        expr_remain = expr_to_eval

    #################
    # Evaluation Phase
    #################
    env = ensure_scope(
        level + 1,
        global_dict=global_dict,
        local_dict=local_dict,
        resolvers=resolvers,
        target=target,
    )

    for assigner, ret in res.items():
        if env.target is not None and assigner is not None:
            target_modified = True

            # if returning a copy, copy only on the first assignment
            if not inplace and first_expr:
                try:
                    target = env.target
                    if isinstance(target, NDFrame):
                        target = target.copy(deep=None)
                    else:
                        target = target.copy()
                except AttributeError as err:
                    raise ValueError("Cannot return a copy of the target") from err
            else:
                target = env.target

            # TypeError is most commonly raised (e.g. int, list), but you
            # get IndexError if you try to do this assignment on np.ndarray.
            # we will ignore numpy warnings here; e.g. if trying
            # to use a non-numeric indexer
            try:
                with warnings.catch_warnings(record=True):
                    # TODO: Filter the warnings we actually care about here.
                    if inplace and isinstance(target, NDFrame):
                        target.loc[:, assigner] = ret
                    else:
                        target[  # pyright: ignore[reportGeneralTypeIssues]
                            assigner
                        ] = ret
            except (TypeError, IndexError) as err:
                raise ValueError("Cannot assign expression output to target") from err

            if not resolvers:
                resolvers = ({assigner: ret},)
            else:
                # existing resolver needs updated to handle
                # case of mutating existing column in copy
                for resolver in resolvers:
                    if assigner in resolver:
                        resolver[assigner] = ret
                        break
                else:
                    resolvers += ({assigner: ret},)

            ret = None
            first_expr = False
            env.target = target

    # evaluate un-optimized exprs
    for e2e in expr_remain:
        # get our (possibly passed-in) scope
        env = ensure_scope(
            level + 1,
            global_dict=global_dict,
            local_dict=local_dict,
            resolvers=resolvers,
            target=target,
        )

        engine = _check_need_fallback(e2e, engine)

        # construct the engine and evaluate the parsed expression
        eng = ENGINES[engine]
        eng_inst = eng(e2e)

        # Temporary engine numexpr fallback check
        # TODO: move it ahead, avoid eval many times
        try:
            ret = eng_inst.evaluate()
        except ValueError:
            engine = 'python'
            eng = ENGINES[engine]
            eng_inst = eng(e2e)
            ret = eng_inst.evaluate()

        if e2e.assigner is None:
            if multi_line:
                raise ValueError(
                    "Multi-line expressions are only valid "
                    "if all expressions contain an assignment"
                )
            if inplace:
                raise ValueError("Cannot operate inplace if there is no assignment")

        # assign if needed
        # put eval result into proper variable identifier
        assigner = e2e.assigner
        if env.target is not None and assigner is not None:
            target_modified = True

            # if returning a copy, copy only on the first assignment
            if not inplace and first_expr:
                try:
                    target = env.target
                    if isinstance(target, NDFrame):
                        target = target.copy(deep=None)
                    else:
                        target = target.copy()
                except AttributeError as err:
                    raise ValueError("Cannot return a copy of the target") from err
            else:
                target = env.target

            # TypeError is most commonly raised (e.g. int, list), but you
            # get IndexError if you try to do this assignment on np.ndarray.
            # we will ignore numpy warnings here; e.g. if trying
            # to use a non-numeric indexer
            try:
                with warnings.catch_warnings(record=True):
                    # TODO: Filter the warnings we actually care about here.
                    if inplace and isinstance(target, NDFrame):
                        target.loc[:, assigner] = ret
                    else:
                        target[  # pyright: ignore[reportGeneralTypeIssues]
                            assigner
                        ] = ret
            except (TypeError, IndexError) as err:
                raise ValueError("Cannot assign expression output to target") from err

            if not resolvers:
                resolvers = ({assigner: ret},)
            else:
                # existing resolver needs updated to handle
                # case of mutating existing column in copy
                for resolver in resolvers:
                    if assigner in resolver:
                        resolver[assigner] = ret
                        break
                else:
                    resolvers += ({assigner: ret},)

            ret = None
            first_expr = False

    # We want to exclude `inplace=None` as being False.
    if inplace is False:
        return target if target_modified else ret
