import numpy as np
import warnings


def poly_fit(x, y, degree, fit="RANSAC"):
    # check if we can use RANSAC
    if fit == "RANSAC":
        try:
            # ignore ImportWarnings in sklearn
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ImportWarning)
                import sklearn.linear_model as sklin
                import sklearn.preprocessing as skpre
        except ImportError:
            warnings.warn(
                "fitting mode 'RANSAC' requires the package sklearn, using"
                + " 'poly' instead",
                RuntimeWarning,
            )
            fit = "poly"

    if fit == "poly":
        return np.polyfit(x, y, degree)
    elif fit == "RANSAC":
        model = sklin.RANSACRegressor(
            sklin.LinearRegression(fit_intercept=False)
        )
        xdat = np.asarray(x)
        if len(xdat.shape) == 1:
            # interpret 1d-array as list of len(x) samples instead of
            # one sample of length len(x)
            xdat = xdat.reshape(-1, 1)
        polydat = skpre.PolynomialFeatures(degree).fit_transform(xdat)
        try:
            model.fit(polydat, y)
            coef = model.estimator_.coef_[::-1]
        except ValueError:
            warnings.warn(
                "RANSAC did not reach consensus, " + "using numpy's polyfit",
                RuntimeWarning,
            )
            coef = np.polyfit(x, y, degree)
        return coef
    else:
        raise ValueError("invalid fitting mode ({})".format(fit))


def rowwise_euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def delay_embedding(data, emb_dim, lag=1):
    """
    Perform a time-delay embedding of a time series

    Args:
      data (array-like):
        the data that should be embedded
      emb_dim (int):
        the embedding dimension
    Kwargs:
      lag (int):
        the lag between elements in the embedded vectors

    Returns:
      emb_dim x m array:
        matrix of embedded vectors of the form
        [data[i], data[i+lag], data[i+2*lag], ... data[i+(emb_dim-1)*lag]]
        for i in 0 to m-1 (m = len(data)-(emb_dim-1)*lag)
    """
    data = np.asarray(data)
    min_len = (emb_dim - 1) * lag + 1
    if len(data) < min_len:
        msg = (
            "cannot embed data of length {} with embedding dimension {} "
            + "and lag {}, minimum required length is {}"
        )
        raise ValueError(msg.format(len(data), emb_dim, lag, min_len))
    m = len(data) - min_len + 1
    indices = np.repeat([np.arange(emb_dim) * lag], m, axis=0)
    indices += np.arange(m).reshape((m, 1))
    return data[indices]


def lyap_r_len(**kwargs):
    """
    Helper function that calculates the minimum number of data points required
    to use lyap_r.

    Note that none of the required parameters may be set to None.

    Kwargs:
      kwargs(dict):
        arguments used for lyap_r (required: emb_dim, lag, trajectory_len and
        min_tsep)

    Returns:
      minimum number of data points required to call lyap_r with the given
      parameters
    """
    # minimum length required to find single orbit vector
    min_len = (kwargs["emb_dim"] - 1) * kwargs["lag"] + 1
    # we need trajectory_len orbit vectors to follow a complete trajectory
    min_len += kwargs["trajectory_len"] - 1
    # we need min_tsep * 2 + 1 orbit vectors to find neighbors for each
    min_len += kwargs["min_tsep"] * 2 + 1
    return min_len
