from endaq.calc.integrate import integrals
from typing import List, Union
import pandas as pd
import numpy as np
from feature_extraction.utils import array_filter


def acc_to_vel_endaq(
    input_array, fs, highpass_cutoff=1, tukey_percent=0.05, zero="mean"
):
    time_array = np.arange(0, len(input_array) / fs, 1 / fs)

    df = pd.DataFrame(data=input_array, index=time_array)

    return integrals(
        df,
        n=1,
        highpass_cutoff=highpass_cutoff,
        tukey_percent=tukey_percent,
        zero=zero,
    )[1].values[:, 0]


def acc_to_displ_endaq(
    input_array, fs, highpass_cutoff=1, tukey_percent=0.05, zero="mean"
):
    time_array = np.arange(0, len(input_array) / fs, 1 / fs)

    df = pd.DataFrame(data=input_array, index=time_array)

    return integrals(
        df,
        n=2,
        highpass_cutoff=highpass_cutoff,
        tukey_percent=tukey_percent,
        zero=zero,
    )[2].values[:, 0]


def acc_to_vel(input_array, fs, convert=False):
    """
    Acceleration to velocity conversion.
    """
    if convert:
        convert = 9.81 / 0.001  # g to mm/s^2
        input_array = convert * input_array

    # input_array.mean() DC component
    zero_mean_input_array = input_array - input_array.mean()
    integral = zero_mean_input_array.cumsum()
    velocity = integral / fs
    velocity_zero_mean = velocity - velocity.mean()

    return velocity_zero_mean


def acc_to_displ(input_array: np.array, fs: int, convert_acc=True) -> np.array:
    """
    Converts acceleration signal to displacement using acc_to_vel.
    """
    return acc_to_vel(
        acc_to_vel(input_array, fs, convert=convert_acc), fs, convert=False
    )


def velocity_transform(
    X: pd.DataFrame,
    signal_column: str,
    sampling_rate_column: str,
    cutoff: Union[List[float], str] = None,
    method: str = "DYX",
    filt_order: int = 3,
    axis: str = "",
) -> pd.DataFrame:
    if method == "DYX":
        velocity_function = acc_to_vel
    elif method == "ENDAQ":
        velocity_function = acc_to_vel_endaq
    else:
        raise ValueError("Invalid method")

    X = X[[signal_column, sampling_rate_column]].copy()

    if cutoff is not None:
        # Filtro aplicado depois da transformação
        df = pd.DataFrame(
            X.apply(
                lambda x: array_filter(
                    velocity_function(x.iloc[0], x.iloc[1]),
                    x.iloc[1],
                    cutoff,
                    filt_order=filt_order,
                ),
                axis=1,
            )
        )
    else:
        df = pd.DataFrame(
            X.apply(lambda x: velocity_function(x.iloc[0], x.iloc[1]), axis=1)
        )

    df.columns = [f"velocity_{axis}"]
    return df
