from typing import List
import numpy as np


def get_fft(
    signal_arr: np.array,
    fs: int,
) -> List[np.array]:
    """
    Returns the one-sided FFT of a signal array and its corresponding frequency vector.
    """

    N = int(len(signal_arr))
    FFT_full = np.fft.fft(signal_arr) / N
    freqVec_full = np.fft.fftfreq(len(FFT_full), 1 / fs)
    j = freqVec_full > 0
    FFT_onesided = FFT_full[j]
    freqVec_onesided = freqVec_full[j]

    return freqVec_onesided, np.abs(2 * FFT_onesided)
