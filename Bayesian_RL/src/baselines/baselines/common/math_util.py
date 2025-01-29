import numpy as np
import scipy.signal


def discount(x, gamma):
    """
    Computes discounted sums along 0th dimension of x.

    inputs
    ------
    x: ndarray
    gamma: float

    outputs
    -------
    y: ndarray with same shape as x, satisfying

        y[t] = x[t] + gamma * x[t+1] + gamma^2 * x[t+2] + ... + gamma^k * x[t+k],
            where k = len(x) - t - 1
    
    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1 - gamma], x[::-1], axis=0)[::-1]


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y - ypred] / Var[y]

    interpretation:
        ev = 0  => might as well have predicted zero
        ev = 1  => perfect prediction
        ev < 0  => worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


