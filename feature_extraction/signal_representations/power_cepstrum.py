import numpy as np
import scipy as sp
import pandas as pd

from scipy.signal import detrend
from scipy.fft import fft, fftfreq


def get_fft(data, n=None, return_onesided=True, remove_mean=True, reduce_size=False):
    """
    Compute the one-dimensional discrete Fourier transform

    Parameters
    ----------
    data : array_like
        Input data.
    n : int, opt
        Number of samples.
    return_onesided : bool, opt
        If True, returns a one-sided spectrum (valid for real data).
    remove_mean : bool, opt
        If True, removes the average signal
    reduce_size : bool, opt
        If True, rounds FFT results to a given number of decimals

    Returns
    -------
    out : float ndarray
        Array of amplitudes of the discrete Fourier transform coeficients.
    """

    # Number of samples
    if n is None:
        n = int(len(data))
    elif n < int(len(data)):
        raise ValueError(
            "Number of samples should be greater than or equal to the original vector length"
        )

    # Remove average signal
    if remove_mean:
        data = data - np.mean(data)

    # FFT
    sp = fft(data, n=n) / n

    # Correction factor in case of zero padding
    sp = sp * n / len(data)

    # Return a one sided spectrum if desired
    if return_onesided:
        out = np.abs(sp[: n // 2])
        out[1:] = out[1:] * 2
    else:
        out = np.abs(sp)

    # Reduce memory usage for front-end
    if reduce_size:
        out = np.round(out, decimals=5)

    return out


def get_fftfreq(samples, fs, return_onesided=True):
    """
    Return the discrete Fourier transform sample frequencies

    Parameters
    ----------
    samples : int
        Number of samples.
    fs : int
        Sampling rate.
    return_onesided : bool, opt
        If True, returns a one-sided spectrum (valid for real data).

    Returns
    -------
    freq : ndarray
        Array of frequencies
    """
    freq = fftfreq(samples, 1 / fs)
    if return_onesided:
        freq = freq[: samples // 2]
    return freq


# def get_cepstrum(data):


def get_cepstrum(
    X: pd.DataFrame,
    signal_column: str,
    sampling_rate_column: str,
) -> pd.DataFrame:
    """
    Return the cepstrum (spectrum of a logarithmic power spectrum).

    Parameters
    ----------
    data : array_like
        Time-domain signal

    Returns
    -------
    cepstrum : ndarray
        Array of cepstral amplitudes
    """

    X = X[[signal_column, sampling_rate_column]].copy()

    # Compute the spectrum
    fft_df = X.apply(lambda x: get_fft(x.iloc[0], remove_mean=False), axis=1)

    # Compute frequency vector
    # freq_vector = get_fftfreq(samples, fs)

    # Return the logarithm of the power spectrum
    log_X = fft_df.apply(lambda x: get_logarithm_spectrum(x**2))

    # Return the spectrum of the logarithmic power spectrum
    cepstrum = log_X.apply(lambda x: np.abs(get_fft(x, remove_mean=False)) ** 2)

    quefrency = X.apply(lambda x: quefrequencies(len(x.iloc[0]), x.iloc[1]), axis=1)

    df = pd.DataFrame([cepstrum.apply(lambda x: x), quefrency.apply(lambda x: x)]).T

    df.columns = ["cepstrum", "quefrency"]

    return df


def get_logarithm_spectrum(X, dB_scale=True, remove_mean=False, remove_trend=False):
    """
    Compute the logarithm of the spectrum

    Parameters
    ----------
    X : ndarray
        Array of spectral amplitudes
    dB_scale : bool, opt
        If True, applies the formula 10*log10(X).
    remove_mean : bool, opt
        If True, it removes the mean value of the logarithmic spectrum
    remove_trend: bool, opt
        If True, it removes the linear trend of the logarithmic spectrum

    Returns
    -------
    log_X : ndarray
        Array of logarithmic spectral amplitudes
    """

    # Substitute zero values in the FFT spectrum to avoid infinite
    # values in the log operation
    MIN_VALUE = 1e-16
    X = np.where(X == 0, MIN_VALUE, X)

    # Compute the logarithm (base 10) of the spectral amplitudes
    if dB_scale:
        log_X = 10 * np.log10(X)
    else:
        log_X = np.log10(X)

    # Remove the mean value of the logarithmic spectrum
    if remove_mean:
        log_X = log_X - np.mean(log_X)

    # Remove the linear trend of the logarithmic spectrum
    if remove_trend:
        log_X = detrend(log_X)

    return log_X


def quefrequencies(samples, fs):
    return get_fftfreq(samples // 2, (samples / 2) / (fs / 2))
