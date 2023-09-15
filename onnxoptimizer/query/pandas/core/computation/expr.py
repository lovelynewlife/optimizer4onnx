from __future__ import annotations

import warnings
from typing import List

from pandas.core import common as com
from pandas.core.generic import NDFrame
from pandas.io.formats import printing

from onnxoptimizer.query.onnx.context import MultiModelContext
from onnxoptimizer.query.pandas.core.computation.ops import is_term
from onnxoptimizer.query.pandas.core.computation.scope import Scope
from onnxoptimizer.query.pandas.core.computation.visitor import PARSERS


def _assign_value(target, assigner, ret, inplace):
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

    return target


class Expr:
    """
    Object encapsulating an expression.

    Parameters
    ----------
    expr : str
    engine : str, optional, default 'numexpr'
    parser : str, optional, default 'pandas'
    env : Scope, optional, default None
    level : int, optional, default 2
    """

    env: Scope
    engine: str
    parser: str

    def __init__(
            self,
            expr,
            engine: str = "numexpr",
            parser: str = "pandas",
            env: Scope | None = None,
            level: int = 0,
    ) -> None:
        self.expr = expr
        self.env = env or Scope(level=level + 1)
        self.engine = engine
        self.parser = parser
        self._visitor = PARSERS[parser](self.env, self.engine, self.parser)
        self.terms = self.parse()

        self.eval_value = None

    @property
    def evaluated(self):
        return self.eval_value is not None

    @property
    def assigner(self):
        return getattr(self._visitor, "assigner", None)

    def assign_value(self, target,  resolvers, inplace):
        assigner = self.assigner
        if target is not None and assigner is not None:
            ret = self.evaluate()
            target = _assign_value(target, assigner, ret, inplace)

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

        return target, resolvers

    def evaluate(self):
        if self.evaluated:
            return self.eval_value
        else:
            return self.terms(self.env)

    def __call__(self):
        return self.evaluate()

    def __repr__(self) -> str:
        return printing.pprint_thing(self.terms)

    def __len__(self) -> int:
        return len(self.expr)

    def parse(self):
        """
        Parse an expression.
        """
        return self._visitor.visit(self.expr)

    @property
    def names(self):
        """
        Get the names in an expression.
        """
        if is_term(self.terms):
            return frozenset([self.terms.name])
        return frozenset(term.name for term in com.flatten(self.terms))


class ComposedExpr:
    env: Scope
    engine: str

    def __init__(
            self,
            engine: str = "onnxruntime",
            env: Scope | None = None,
            level: int = 0,
            terms: MultiModelContext | None = None,
            assigners: List[str] | None = None
    ) -> None:
        self.env = env or Scope(level=level + 1)
        self.engine = engine
        self.assigners = assigners
        self.terms = terms

        self.eval_value = None

    @property
    def assigner(self):
        return self.assigners

    @property
    def evaluated(self):
        return self.eval_value is not None

    def evaluate(self):
        if self.evaluated:
            return self.eval_value
        else:
            return self.terms()

    def __call__(self):
        return self.evaluate()

    def assign_value(self, target,  resolvers, inplace):
        assigners = self.assigner
        if target is not None and assigners is not None:
            ret = self.evaluate()
            for assigner in assigners:
                if assigner is not None:
                    # TODO: another way to dispatch composed inference value
                    res = ret[assigner]

                    target = _assign_value(target, assigner, res, inplace)

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

        return target, resolvers

    def __repr__(self) -> str:
        return printing.pprint_thing(self.terms)
