import pandas as pd
from typing import List, Union
from feature_extraction.utils.array_filter import array_filter


def bandpass_filter_transform(
    X: pd.DataFrame,
    signal_column: str,
    sampling_rate_column: str,
    cutoff: Union[List[float], str] = None,
    filt_order: int = 3,
) -> pd.DataFrame:
    if cutoff is None:  # No filter
        return X[[signal_column]]
    else:
        X = X[[signal_column, sampling_rate_column]].copy()
        return pd.DataFrame(
            X.apply(
                lambda x: array_filter(
                    x.iloc[0], x.iloc[1], cutoff, filt_order=filt_order
                ),
                axis=1,
            )
        )
