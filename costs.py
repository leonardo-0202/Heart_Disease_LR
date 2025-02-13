# -*- coding: utf-8 -*-
"""Functions used to compute the loss of error functions."""

import numpy as np

def compute_loss_MSE(y: np.ndarray,
                     tx: np.ndarray,
                     w: np.ndarray) -> float:
    """Calculate the loss using either MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # Error vector
    e = y - tx @ w
    # N = y.shape[0]
    return 1/(2*y.shape[0]) * e @ e 

def compute_loss_MAE(y: np.ndarray,
                     tx: np.ndarray,
                     w: np.ndarray) -> float:
    """Compute a subgradient of the MAE at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    return 1/y.shape[0] * (y - tx @ w)