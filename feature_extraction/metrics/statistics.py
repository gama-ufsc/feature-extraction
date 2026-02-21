""" 
Module to define statistical metrics.

This module contains functions to calculate various statistical metrics for a given input vector.

Functions:
----------
rms(input_vector: np.ndarray) -> float:
    Calculates the root mean square of the input vector.

pk_pk(input_vector: np.ndarray) -> float:
    Calculates the peak-to-peak value of the input vector.

kurtosis(input_vector: np.ndarray) -> float:
    Calculates the kurtosis of the input vector.

crest_factor(input_vector: np.ndarray) -> float:
    Calculates the crest factor of the input vector.

crest_factor_plus(input_vector: np.ndarray) -> float:
    Calculates the crest factor plus of the input vector.

skewness(input_vector: np.array) -> float:
    Calculates the skewness of a given input vector.

shape_factor(input_vector: np.ndarray) -> float:
    Calculates the shape factor of the input vector.

clearance_factor(input_vector: np.ndarray) -> float:
    Calculates the clearance factor of the input vector.

impulse_factor(input_vector: np.ndarray) -> float:
    Calculates the impulse factor of the input vector.
"""
import numpy as np
from scipy import stats


def rms(input_vector: np.ndarray) -> float:
    """
    Calculates the root mean square of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the root mean square is to be calculated.

    Returns:
    --------
    float:
        Root mean square of the input vector.
    """
    input_vector = np.asarray(input_vector)
    input_squared = input_vector**2
    mean = np.mean(input_squared)
    rms = np.sqrt(mean)

    return rms


def pk_pk(input_vector: np.ndarray) -> float:
    """
    Calculates the peak-to-peak value of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the peak-to-peak value is to be calculated.

    Returns:
    --------
    float:
        Peak-to-peak value of the input vector.
    """
    input_vector = np.asarray(input_vector)
    pkpk = input_vector.max() - input_vector.min()

    return pkpk


def kurtosis(input_vector: np.ndarray) -> float:
    """
    Calculates the kurtosis of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the kurtosis is to be calculated.

    Returns:
    --------
    float:
        Kurtosis of the input vector.
    """
    input_vector = np.asarray(input_vector)
    kurt = stats.kurtosis(input_vector, fisher=False)
    return kurt if not np.isnan(kurt) else 0


def crest_factor(input_vector: np.ndarray) -> float:
    """
    Calculates the crest factor of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the crest factor is to be calculated.

    Returns:
    --------
    float:
        Crest factor of the input vector.
    """
    input_vector = np.asarray(input_vector)
    my_pkvalue = np.max(np.abs(input_vector))
    my_rms = rms(input_vector)
    cf = my_pkvalue / my_rms if my_rms != 0 else 0.0

    return cf


def skewness(input_vector: np.array) -> float:
    """
    Calculates the skewness of a given input vector.

    Parameters:
    -----------
    input_vector (np.array): The input vector for which skewness is to be calculated.

    Returns:
    --------
    float: The skewness of the input vector.
    """
    input_vector = np.asarray(input_vector)
    skew = stats.skew(input_vector)
    return skew if not np.isnan(skew) else 0


def shape_factor(input_vector: np.ndarray) -> float:
    """
    Calculates the shape factor of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the shape factor is to be calculated.

    Returns:
    --------
    float:
        Shape factor of the input vector.
    """
    input_vector = np.asarray(input_vector)
    my_rms = rms(input_vector)
    abs_value_mean = np.mean(np.abs(input_vector))
    sf = my_rms / abs_value_mean if abs_value_mean != 0 else 0.0

    return sf


def clearance_factor(input_vector: np.ndarray) -> float:
    """
    Calculates the clearance factor of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the clearance factor is to be calculated.

    Returns:
    --------
    float:
        Clearance factor of the input vector.
    """
    input_vector = np.asarray(input_vector)
    my_pkvalue = np.max(np.abs(input_vector))
    cl_denominator = np.mean(np.sqrt(np.abs(input_vector))) ** 2
    cl = my_pkvalue / cl_denominator if cl_denominator != 0 else 0.0

    return cl


def impulse_factor(input_vector: np.ndarray) -> float:
    """
    Calculates the impulse factor of the input vector.

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the impulse factor is to be calculated.

    Returns:
    --------
    float:
        Impulse factor of the input vector.
    """
    input_vector = np.asarray(input_vector)
    my_pkvalue = np.max(np.abs(input_vector))
    abs_value_mean = np.mean(np.abs(input_vector))
    imp = my_pkvalue / abs_value_mean if abs_value_mean != 0 else 0.0

    return imp


def hjorth_activity(input_vector: np.ndarray) -> float:
    """
    Calculates the activity of the input vector using Hjorth parameters.
    Calculation: https://en.wikipedia.org/wiki/Hjorth_parameters

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the activity is to be calculated.

    Returns:
    --------
    float:
        Activity of the input vector.
    """

    activity = np.var(input_vector)

    return activity


def hjorth_mobility(input_vector: np.ndarray) -> float:
    """
    Calculates the mobility of the input vector using Hjorth parameters.
    Calculation: https://en.wikipedia.org/wiki/Hjorth_parameters

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the mobility is to be calculated.

    Returns:
    --------
    float:
        Mobility of the input vector.
    """

    mobility = np.sqrt(np.var(np.diff(input_vector))) / np.sqrt(
        np.var(input_vector)
    )

    return mobility


def hjorth_complexity(input_vector: np.ndarray) -> float:
    """
    Calculates the complexity of the input vector using Hjorth parameters.
    Calculation: https://en.wikipedia.org/wiki/Hjorth_parameters

    Parameters:
    -----------
    input_vector: np.ndarray
        Input vector for which the complexity is to be calculated.

    Returns:
    --------
    float:
        Complexity of the input vector.
    """

    complexity = hjorth_mobility(np.diff(input_vector)) / hjorth_mobility(
        input_vector
    )

    return complexity
