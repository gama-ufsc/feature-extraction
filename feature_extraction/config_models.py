from pydantic import BaseModel, field_validator, model_validator
from typing import Callable, Dict, Any, Optional, List, Union
import pandas as pd
from feature_extraction.validate_functions import (
    validate_return_type,
    validate_required_params,
    validate_param_types,
)


class SignalRepresentationConfig(BaseModel):
    """
    Configuration model for representing a signal processing function and its parameters.

    Parameters
    ----------
        func (Callable[..., pd.DataFrame]): A signal processing function that returns a pandas DataFrame.
        params (Dict[str, Any]): A dictionary of parameters to pass to the function.
        name (Optional[str]): An optional name for this configuration.
    """

    func: Callable[..., pd.DataFrame]
    params: Dict[str, Any] = {}
    name: Optional[str] = None

    @field_validator("func")
    def check_func(cls, func):
        """Validates that the function returns a pandas DataFrame."""
        validate_return_type(func, pd.DataFrame)
        return func

    @model_validator(mode="after")
    def check_params(self):
        """
        Validates that all required parameters are provided and that provided
        parameter types match the function annotations.
        """
        func = self.func
        params = self.params

        if func:
            validate_required_params(func, params, "X")
            validate_param_types(func, params)

        return self

    # Custom validator to handle partial functions
    @field_validator("func", mode="before")
    def validate_func(cls, v):
        # Custom logic to handle partial functions
        if hasattr(v, "func"):
            v = v.func
        return v


class MetricConfig(BaseModel):
    """
    Configuration model for representing a metric function and its parameters.

    Parameters
    ----------
        func (Callable[..., pd.DataFrame]): A metric function that returns a float or list.
        params (Optional[Dict[str, Any]]): A dictionary of parameters to pass to the function.
        name (Optional[str]): An optional name for this configuration.
    """

    func: Callable[..., Union[float, List]]
    params: Optional[Dict[str, Any]] = {}
    name: Optional[str] = None

    # If name is not provided, use the function name
    @model_validator(mode="after")
    def set_name(self):
        if self.name is None:
            self.name = self.func.__name__

        return self

    # Custom validator to handle partial functions
    @field_validator("func", mode="before")
    def validate_func(cls, v):
        # Custom logic to handle partial functions
        if hasattr(v, "func"):
            v = v.func
        return v


class TransformConfig(BaseModel):
    """
    Configuration model for instantiate a Transform class for feature extraction.

    Parameters
    ----------
        signal_representation (SignalRepresentationConfig): A configuration for the signal representation.
        metrics (List[MetricConfig]): A list of configurations for the metrics to apply to the signal representation.
    """

    signal_representation: SignalRepresentationConfig
    metrics: List[MetricConfig]
