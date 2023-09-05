import copy
from typing import Callable

from gorilla import Settings, _get_base, Patch, get_decorator_data


def callable_patch(destination: object,
                   name: str = None,
                   settings: Settings = None) -> Callable[[Callable], Callable]:
    def decorator(wrapped: Callable):
        base = _get_base(wrapped)
        name_ = base.__name__ if name is None else name
        settings_ = copy.deepcopy(settings)
        patch = Patch(destination, name_, wrapped, settings=settings_)
        data = get_decorator_data(base, set_default=True)
        data.patches.append(patch)
        return wrapped

    return decorator
