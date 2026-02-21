import numpy as np
from typing import List


def rahmonic_peaks(
    input_vector: np.ndarray,
    quefrency_vector: np.ndarray,
    fundamental_frequency: float,
    rahmonics: List[float] = [1.0, 2.0, 3.0],
    pct_change=0.05,
    n_points=1,
    rpm=False,
    **kwargs
) -> List[float]:
    """F
    unction to compute the maximum value of the input vector within a frequency range.

    Parameters:
    -----------
        input_vector: np.ndarray - input vector in frequency domain
        quefrency_vector: np.ndarray - frequency vector
        fundamental_frequency: float - frequency of the rahmonic
        rahmonics: List[float] - list of rahmonics to compute
        pct_change: float - percentage change in frequency range
        n_points: int - number of points to include in the frequency range

    Returns:
    --------
        float - maximum value of the input vector within the specified frequency range
    """

    if rpm:
        fundamental_frequency /= 60

    rahmonics_result = []
    for harmonic in rahmonics:
        # harmonic_frequency = harmonic * fundamental_frequency
        rahmonic_quefrency = harmonic * (1 / fundamental_frequency)

        lower_quefrency = rahmonic_quefrency * (1 - pct_change)
        upper_quefrency = rahmonic_quefrency * (1 + pct_change)

        indexs = np.where(
            (quefrency_vector >= lower_quefrency)
            & (quefrency_vector <= upper_quefrency)
        )[0]

        if len(indexs) == 0:
            if lower_quefrency >= quefrency_vector[0]:
                lower_index = max(np.where(quefrency_vector <= lower_quefrency)[0])
            else:
                lower_index = 0

            if upper_quefrency <= quefrency_vector[0]:
                upper_index = n_points
            elif upper_quefrency <= quefrency_vector[-1]:
                upper_index = min(np.where(quefrency_vector >= upper_quefrency)[0])
            else:
                upper_index = len(input_vector) - 1

            rahmonics_result.append(np.max(input_vector[lower_index:upper_index]))
        else:
            rahmonics_result.append(np.max(input_vector[indexs]))

    return rahmonics_result
