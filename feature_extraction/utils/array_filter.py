from scipy import signal
import numpy as np


def bandpass_filter(
    input_signal, N, cutoff, fs, btype="bandpass", analog=False, output="sos"
):
    """
    This function is designed to apply a butterworth filter in a given input signal,
    where the main input variables are:

    input_signal:array_like
    The signal to be filtered (it must be a array)

    N: int or array_like

    cutoff: array_like or list
    Cutoff frequency(ies) in Hz (int). Note that a scalar or length-2
    sequence giving the critical frequencies.

    fs : float
    The sampling frequency of the digital system.

    btype, analog, output: same parameters describe in scipy.signal.butter.

    check_response: boolean
    Plot the frequency response of a digital filter.

    Returns
    -------

    output_signal: ndarray
    Filtered signal.
    """

    if output == "zpk":
        raise Exception(
            'Zero and poles output is not implemented in this version! \
                        You must use "ba" or "sos" output.'
        )

    nyq = 0.5 * fs
    Wn = np.asarray(cutoff, dtype=np.float64) / nyq

    butter_coef = signal.butter(
        N, Wn, btype=btype, analog=analog, output=output
    )

    if output == "sos":
        return signal.sosfiltfilt(
            butter_coef, input_signal, padlen=int(N / 2), padtype="even"
        )

    elif output == "ba":
        return signal.filtfilt(butter_coef[0], butter_coef[1], input_signal)


def array_filter(array, fs, cutoff, filt_order=3):
    """
    Apply bandpass or highpass filter (if necessary) to the array.
    """
    lowcut = float(cutoff[0])
    highcut = float(cutoff[1])
    nyq = fs * 0.5
    Wn_lowcut = lowcut / nyq
    Wn_highcut = highcut / nyq

    if (0 < Wn_lowcut < 1) and (0 < Wn_highcut < 1):
        filtered_array = bandpass_filter(
            array, filt_order, [lowcut, highcut], fs, btype="bandpass"
        )
    elif (0 < Wn_lowcut < 1) and (Wn_highcut >= 1):
        filtered_array = bandpass_filter(
            array, filt_order, lowcut, fs, btype="highpass"
        )
    elif (Wn_lowcut <= 0) and (0 < Wn_highcut < 1):
        filtered_array = bandpass_filter(
            array, filt_order, highcut, fs, btype="lowpass"
        )
    else:
        filtered_array = np.zeros(array.shape)

    return filtered_array
