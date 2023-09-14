from typing import Any

import pandas
from pandas import DataFrame
from pandas.util._validators import validate_bool_kwarg

from onnxoptimizer.query.pandas.core.computation.eval import pandas_eval
from onnxoptimizer.query.util import callable_patch

LEVEL_OFFSET_1 = 1


@callable_patch(pandas)
def predict_eval(expr: str,
                 parser: str = "pandas",
                 engine: str | None = 'python',
                 local_dict: Any = None,
                 global_dict: Any = None,
                 resolvers: Any = (),
                 level: int = 0,
                 target: Any = None,
                 inplace: bool = False,
                 enable_opt: bool = True):
    return pandas_eval(expr, parser=parser, engine=engine,
                       local_dict=local_dict, global_dict=global_dict,
                       resolvers=resolvers, level=level+LEVEL_OFFSET_1,
                       target=target, inplace=inplace, enable_opt=enable_opt)


@callable_patch(DataFrame, name="predict_eval")
def df_predict_eval(self, expr: str, *, inplace: bool = False, **kwargs) -> Any | None:
    inplace = validate_bool_kwarg(inplace, "inplace")
    kwargs["level"] = kwargs.pop("level", 0) + 1
    index_resolvers = self._get_index_resolvers()
    column_resolvers = self._get_cleaned_column_resolvers()
    resolvers = column_resolvers, index_resolvers
    if "target" not in kwargs:
        kwargs["target"] = self
    kwargs["resolvers"] = tuple(kwargs.get("resolvers", ())) + resolvers
    return predict_eval(expr, inplace=inplace, **kwargs)


@callable_patch(DataFrame)
def predict_filter(self, predicate: str, *, inplace: bool = False, **kwargs) -> DataFrame | None:
    inplace = validate_bool_kwarg(inplace, "inplace")
    if not isinstance(predicate, str):
        msg = f"expr must be a string to be evaluated, {type(predicate)} given"
        raise ValueError(msg)
    kwargs["level"] = kwargs.pop("level", 0) + 1
    kwargs["target"] = None
    res = df_predict_eval(self, predicate, **kwargs)

    try:
        result = self.loc[res]
    except ValueError:
        # when res is multi-dimensional loc raises, but this is sometimes a
        # valid query
        result = self[res]

    if inplace:
        self._update_inplace(result)
        return None
    else:
        return result
