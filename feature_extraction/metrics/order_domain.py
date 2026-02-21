import numpy as np
from typing import List

def spectrum_peak_ratio(
        input_vector: np.ndarray,
        order_vector: np.ndarray,
        defect_order: float,
        pct_change: float = 0.05,
        n_points: int = 1,
        **kwargs
) -> float:
    """Function to compute the maximum in frequency domain.

    Parameters:
    -----------
        input_vector: np.ndarray - input vector in frequency domain

    Returns:
    --------
        float - maximum value in the input vector
    """
    def_ind=np.arange(0,order_vector[-1],defect_order)

    peak_defects=[]
    for i in def_ind:
        # lower_order = i * (1 - pct_change)
        # upper_order = i * (1 + pct_change)

        lower_order = i - (pct_change*defect_order)
        upper_order = i + (pct_change*defect_order)

        indexs = np.where((order_vector >= lower_order) & (order_vector <= upper_order))[0]

        if len(indexs) == 0:
            if lower_order >= order_vector[0]:
                lower_index = max(np.where(order_vector <= lower_order)[0])
            else:
                lower_index = 0

            if upper_order <= order_vector[0]:
                upper_index = n_points
            elif upper_order <= order_vector[-1]:
                upper_index = min(np.where(order_vector >= upper_order)[0])
            else:
                upper_index = len(input_vector) - 1

        else:
            lower_index = np.maximum(0, min(indexs) - n_points)
            upper_index = np.minimum(
                len(input_vector) - 1, max(indexs) + n_points
            )

        peak_defects.append(np.max(np.abs(input_vector[lower_index:upper_index])))
    K=len(input_vector)
    sum_peaks=np.sum(peak_defects)
    sum_lines=np.sum(input_vector)

    return ((K*sum_peaks)/(sum_lines))

def harmonic_peak(
    input_vector: np.ndarray,
    order_vector: np.ndarray,
    defect_order: float,
    harmonics: List[float] = [1.0, 2.0, 3.0],
    pct_change: float = 0.05,
    n_points: int = 1,
    **kwargs
) -> List[float]:
    """Function to compute the maximum value of the input vector within a frequency range.

    Parameters:
    -----------
        input_vector: np.ndarray - input vector in frequency domain
        freqVec: np.ndarray - frequency vector
        fundamental_freq: float - frequency of the harmonic
        bearing_freq: float - frequency of the bearing (if applicable)
        harmonics: List[float] - list of harmonics to compute
        pct_change: float - percentage change in frequency range
        n_points: int - number of points to include in the frequency range

    Returns:
    --------
        float - maximum value of the input vector within the specified frequency range
    """

    harmonics_result = []
    for harmonic in harmonics:
        harmonic_freq = harmonic * defect_order

        lower_order = harmonic_freq - (pct_change*defect_order)
        upper_order = harmonic_freq + (pct_change*defect_order)

        indexs = np.where((order_vector >= lower_order) & (order_vector <= upper_order))[0]

        if len(indexs) == 0:
            if lower_order >= order_vector[0]:
                lower_index = max(np.where(order_vector <= lower_order)[0])
            else:
                lower_index = 0

            if upper_order <= order_vector[0]:
                upper_index = n_points
            elif upper_order <= order_vector[-1]:
                upper_index = min(np.where(order_vector >= upper_order)[0])
            else:
                upper_index = len(input_vector) - 1

        else:
            lower_index = np.maximum(0, min(indexs) - n_points)
            upper_index = np.minimum(
                len(input_vector) - 1, max(indexs) + n_points
            )

        harmonics_result.append(
            np.max(np.abs(input_vector[lower_index:upper_index]))
        )

    return harmonics_result