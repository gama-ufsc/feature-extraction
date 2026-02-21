import numpy as np
from scipy import stats
from typing import List
from scipy.interpolate import InterpolatedUnivariateSpline


def spectral_peak(input_vector: np.ndarray) -> float:
    """Function to compute the maximum in frequency domain.

    Parameters:
    -----------
        input_vector: np.ndarray - input vector in frequency domain

    Returns:
    --------
        float - maximum value in the input vector
    """
    return np.max(np.abs(input_vector))


def find_peaks_v2(
    input_vector: np.ndarray,
    freqVec: np.ndarray,
    fundamental_freq: float,
    bearing_freq: float = None,
    harmonics: List[float] = [1.0, 2.0, 3.0],
    pct_change: float = 0.05,
    n_points: int = 10,
    rpm: bool = False,
    **kwargs,
) -> List[float]:
    if rpm:
        fundamental_freq /= 60
    else:
        if bearing_freq is not None:
            raise ValueError("Bearing frequency can only be specified if rpm is True")

    harmonic_freq = fundamental_freq
    if bearing_freq is not None:
        harmonic_freq = bearing_freq * harmonic_freq

    delta_f = freqVec[1] - freqVec[0]

    harmonic_results = []

    lower_freq = (harmonic_freq * (1 - pct_change)) - ((n_points // 2) * delta_f)
    upper_freq = (harmonic_freq * (1 + pct_change)) + (n_points * delta_f)

    indexs = np.where((freqVec >= lower_freq) & (freqVec <= upper_freq))[0]

    lower_bound = np.maximum(0, min(indexs))
    upper_bound = np.minimum(max(indexs), (len(freqVec) - 1))

    peak_index = np.argmax(input_vector[lower_bound:upper_bound])
    peak_freq = freqVec[peak_index + lower_bound]

    fundamental_harmonic_freq = peak_freq

    for harmonic in harmonics:
        harmonic_freq = fundamental_harmonic_freq * harmonic

        lower_freq = (harmonic_freq * (1 - pct_change)) - (delta_f)
        upper_freq = (harmonic_freq * (1 + pct_change)) + (2 * delta_f)

        indexs = np.where((freqVec >= lower_freq) & (freqVec <= upper_freq))[0]

        lower_bound = np.maximum(0, min(indexs))
        upper_bound = np.minimum(max(indexs), (len(freqVec) - 1))

        peak_value = np.max(input_vector[lower_bound:upper_bound])

        harmonic_results.append(peak_value)

    return harmonic_results