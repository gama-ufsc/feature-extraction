from inspect import signature
from typing import (
    Callable,
    Dict,
    Any,
    get_type_hints,
)


def validate_return_type(func: Callable, type: Any) -> None:
    annotations = get_type_hints(func)
    if annotations.get("return") is not type:
        raise TypeError(f"Function `{func.__name__}` must return {type}.")


def validate_required_params(
    func: Callable, params: Dict[str, Any], transform_param
) -> None:
    sig = signature(func)
    valid_params = set(sig.parameters.keys())
    provided_params = set(params.keys())

    # Check for missing required parameters
    required_params = {
        name
        for name, param in sig.parameters.items()
        if (param.default is param.empty) and (param.name != transform_param)
    }
    missing_params = required_params - provided_params
    if missing_params:
        raise ValueError(
            f"Function `{func.__name__}` missing required parameters: {', '.join(missing_params)} \n"
        )

    # Check for invalid parameters
    invalid_params = provided_params - valid_params
    if invalid_params:
        raise ValueError(
            f"Function `{func.__name__}` received invalid parameters: {', '.join(invalid_params)} \n"
        )


def validate_param_types(func: Callable, params: Dict[str, Any]) -> None:
    sig = signature(func)
    for name, param in sig.parameters.items():
        # Check for missing type annotations
        if param.annotation is param.empty:
            raise TypeError(
                f"Function `{func.__name__}` parameter `{name}` missing type annotation"
            )
