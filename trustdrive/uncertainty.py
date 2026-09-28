"""Ensemble uncertainty utilities (epistemic uncertainty of the perception).

Given ``M`` predictions of the same quantity from an ensemble of models we
report the ensemble mean and the (unbiased) empirical covariance::

    xbar   = 1/M sum_i x_i
    Sigma  = 1/(M-1) sum_i (x_i - xbar)(x_i - xbar)^T
"""
from __future__ import annotations

import numpy as np


def ensemble_stats(predictions: np.ndarray):
    """Mean and covariance of an ensemble.

    Parameters
    ----------
    predictions : array ``(M, d)`` - ``M`` member predictions of dimension ``d``.

    Returns
    -------
    mean : ``(d,)``
    cov  : ``(d, d)`` empirical covariance (epistemic uncertainty).
    """
    P = np.atleast_2d(np.asarray(predictions, dtype=float))
    M = P.shape[0]
    mean = P.mean(axis=0)
    if M < 2:
        return mean, np.zeros((P.shape[1], P.shape[1]))
    d = P - mean
    cov = (d.T @ d) / (M - 1)
    return mean, cov


def total_variance(cov: np.ndarray) -> float:
    """Scalar summary of uncertainty (trace of the covariance)."""
    return float(np.trace(np.atleast_2d(cov)))
