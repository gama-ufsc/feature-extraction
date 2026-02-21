import pandas as pd
import numpy as np
from feature_extraction.utils import array_filter
from typing import List, Union


def signal_time_domain(
    X: pd.DataFrame,
    signal_column: str,
    sampling_rate_column: str,
    cutoff: Union[List[float], str] = None,
    filt_order: int = 3,
    return_time_vector: bool = False,
) -> pd.DataFrame:
    """Dataframe containing the signal in the time domain.

    Args:
        X (pd.DataFrame): dataframe with metadata and signal of each measurement and accelerometer.
        signal_column (str): name of the column containing the signal.
        sampling_rate_column (str): name of the column containing the sampling rate.
        cutoff (_type_, optional): filter information. Defaults to None.
        filt_order (int, optional): order of the implemented filter. Defaults to 3.
        return_time_vector (bool, optional): if True also returns time vector of the signal. Defaults to False.

    Returns:
        pd.DataFrame: time signal representation and IF return_time_vector is True, also time vector.
    """

    X = X[[signal_column, sampling_rate_column]].copy()
    if cutoff is None:  # No filter
        time_amplitude = X[signal_column].apply(lambda x: np.array(x))

    else:
        time_amplitude = X.apply(
            lambda x: array_filter(
                x.iloc[0], x.iloc[1], cutoff, filt_order=filt_order
            ),
            axis=1,
        )

    df = pd.DataFrame([time_amplitude]).T
    df.columns = ["time_amplitude"]

    if return_time_vector:
        time_vector = np.arange(
            0,
            len(time_amplitude[0]) / X[sampling_rate_column][0],
            1 / X[sampling_rate_column][0],
        )
        df = pd.DataFrame([time_amplitude, time_vector]).T
        df.columns = ["time_amplitude", "time_vector [s]"]

    return df
