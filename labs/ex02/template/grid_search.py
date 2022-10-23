# -*- coding: utf-8 -*-
"""Exercise 2.

Grid Search
"""

import numpy as np
from costs import compute_loss


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, grid_w0, grid_w1):
    """Algorithm for grid search.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        grid_w0: numpy array of shape=(num_grid_pts_w0, ). A 1D array containing num_grid_pts_w0 values of parameter w0 to be tested in the grid search.
        grid_w1: numpy array of shape=(num_grid_pts_w1, ). A 1D array containing num_grid_pts_w1 values of parameter w1 to be tested in the grid search.
        
    Returns:
        losses: numpy array of shape=(num_grid_pts_w0, num_grid_pts_w1). A 2D array containing the loss value for each combination of w0 and w1
    """

    losses = np.zeros((len(grid_w0), len(grid_w1)))

    for idx_0, w0 in enumerate(grid_w0):
        for idx_1, w1 in enumerate(grid_w1):
            losses[idx_0, idx_1] = compute_loss(y, tx, np.array([w0, w1]))
            
    return losses


# # Generate the grid of parameters to be swept
# grid_w0, grid_w1 = generate_w(num_intervals=100)

# # Start the grid search
# start_time = datetime.datetime.now()
# grid_losses = grid_search(y, tx, grid_w0, grid_w1)

# # Select the best combinaison
# loss_star, w0_star, w1_star = get_best_parameters(grid_w0, grid_w1, grid_losses)
# end_time = datetime.datetime.now()
# execution_time = (end_time - start_time).total_seconds()

# # Print the results
# print("Grid Search: loss*={l}, w0*={w0}, w1*={w1}, execution time={t:.3f} seconds".format(
#       l=loss_star, w0=w0_star, w1=w1_star, t=execution_time))

# # Plot the results
# fig = grid_visualization(grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight)
# fig.set_size_inches(10.0,6.0)
# fig.savefig("grid_plot")  # Optional saving