# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # least squares
    opt_w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    
    N = y.shape[0]
    e =  y - tx.dot(opt_w)
    loss_mse = (1/(2*N))*e.dot(e.T)

    
    return opt_w, loss_mse

