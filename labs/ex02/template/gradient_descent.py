# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""

from costs import *

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """

    # compute gradient vector
    N = y.shape[0]
    x_t = tx.T
    e = y-tx.dot(w)
    det_L = (-1/N)*(x_t.dot(e))
    
    return det_L


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):

        # compute gradient and loss
        gradient = compute_gradient(y, tx, initial_w)
        loss = compute_loss(y, tx, w)
        
        # update w by gradient
        w = w - gamma*gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        # print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #       bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws, losses 
