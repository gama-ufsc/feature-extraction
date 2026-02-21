from scipy import signal
import numpy as np
from typing import List, Union
import pandas as pd
from feature_extraction.utils import array_filter, get_fft, kurtogram_bandpass


def get_envelope(
    X: pd.DataFrame,
    signal_column: str,
    sampling_rate_column: str,
    domain: str,
    cutoff: Union[List[float], str] = [100, 500],
    order: int = 12,
) -> pd.DataFrame:
    """
    Compute the envelope of a signal.

    Parameters:
    -----------
    X : pd.DataFrame
        Dataframe containing the signal and its sampling rate.
    signal_column : str
        Name of the column containing the signal.
    sampling_rate_column : str
        Name of the column containing the sampling rate.
    domain : str
        Define if the return is in time domain or frequency domain
    cutoff : Union[List[float], str], optional
        Cutoff frequency for the filter. If 'kurtogram', the cutoff frequency is computed using the kurtogram method,
        by default [100, 500]
    order : int, optional
        Order of the filter, by default 12

    Returns:
    --------
    pd.DataFrame
        Dataframe containing the envelope of the signal.
    """
    X = X[[signal_column, sampling_rate_column]].copy()

    if cutoff == "kurtogram":
        kurtogram_values = X.apply(
            lambda x: kurtogram_bandpass(x.iloc[0], x.iloc[1]), axis=1
        )
        X["kurtogram_cutoff"] = kurtogram_values
        filtered_signal = X.apply(
            lambda row: array_filter(
                row.iloc[0],
                row.iloc[1],
                row["kurtogram_cutoff"],
                filt_order=order,
            ),
            axis=1,
        )
    elif cutoff is None:
        filtered_signal = X[signal_column]
    else:
        filtered_signal = X.apply(
            lambda row: array_filter(
                row.iloc[0], row.iloc[1], cutoff, filt_order=order
            ),
            axis=1,
        )

    temporal_envelope = filtered_signal.apply(
        lambda x: np.abs(signal.hilbert(x))
    )

    if domain == 'time':
        envelope=pd.DataFrame()
        envelope['timeEnvelope']=temporal_envelope
    else:

        freqVec_specEnvelope = pd.DataFrame(
            [temporal_envelope, X.apply(lambda row: row.iloc[1], axis=1)]
        ).T.apply(lambda x: get_fft(x.iloc[0], x.iloc[1]), axis=1)

        envelope = pd.DataFrame(
            [
                freqVec_specEnvelope.apply(lambda x: x[1]),
                freqVec_specEnvelope.apply(lambda x: x[0]),
            ]
        ).T

        envelope.columns = ["specEnvelope", "freqVec"]

    return envelope
