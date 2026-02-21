import pandas as pd
from feature_extraction.utils import array_filter, get_fft
from typing import List, Union


def fft_transform(
    X: pd.DataFrame,
    signal_column: str,
    sampling_rate_column: str,
    cutoff: Union[List[float], str] = None,
    filt_order: int = 3,
) -> pd.DataFrame:
    """
    Computes the Fast Fourier Transform (FFT) of a signal.

    Parameters:
    -----------
        X (pd.DataFrame): Input DataFrame containing the signal and sampling rate columns.
        signal_column (str): Name of the column containing the signal.
        sampling_rate_column (str): Name of the column containing the sampling rate.
        cutoff (float, optional): Cutoff frequency for the low-pass filter. Defaults to None.
        filt_order (int, optional): Order of the Butterworth filter. Defaults to 3.

    Returns:
    --------
        pd.DataFrame: DataFrame containing the FFT and frequency vector columns.
    """
    X = X[[signal_column, sampling_rate_column]].copy()
    if cutoff is None:  # No filter
        fft = X.apply(lambda x: get_fft(x.iloc[0], x.iloc[1]), axis=1)

    else:
        fft = X.apply(
            lambda x: get_fft(
                array_filter(x.iloc[0], x.iloc[1], cutoff, filt_order=filt_order),
                x.iloc[1],
            ),
            axis=1,
        )

    df = pd.DataFrame([fft.apply(lambda x: x[1]), fft.apply(lambda x: x[0])]).T

    df.columns = ["fft", "freqVec"]
    return df
