import jax.numpy as jnp
import numpy as np


# The z_i function
def calc_augmented_depvar(Y, X, gamma):
    """Calculates the augmented dependent variable for the PDR estimation."""
    Xgamma = X @ gamma

    assert (
        Xgamma.shape[0] == X.shape[0]
    ), "Xgamma should be of length equal to {}".format(X.shape[0])
    assert len(Xgamma.shape) == 1, "Xgamma should be a 1d array"
    assert len(Y.shape) == 1, "Y should be a 1d array"

    Z = (Y - jnp.exp(Xgamma)) / jnp.exp(Xgamma) + Xgamma
    return Z


def calc_weight_diag(X, gamma):
    """Calculates the weight for the PDR estimation."""

    return jnp.exp(X @ gamma)


def estimate_pdr(Y, X, g0, max_iter=100, tol=1e-6):
    """Estimates the PDR regression."""
    # Initialize the loop variables
    diff = 1
    iter = 0

    ests = np.zeros((max_iter, g0.shape[0])) + np.nan
    while diff > tol and iter < max_iter:
        # Set the initial value
        g = g0
        # calc the weight matrix
        #W = jnp.diag(calc_weight_diag(X, g))
        W = calc_weight_diag(X, g)
        # calc the augmented dependent variable
        Z = calc_augmented_depvar(Y, X, g)

        # pdr regression
        xw = jnp.multiply(X.T, W)

        g0 = jnp.linalg.solve(xw @ X, xw @ Z)
        ests[iter, :] = g0
        diff = jnp.abs(g0 - g).max()
        print(diff)
        iter += 1

    return g0, iter, diff, ests


def estimate_pdr_experimental(Y, X, g0, max_iter=100, tol=1e-6):
    """Estimates the PDR regression."""
    # Initialize the loop variables
    diff = 1
    iter = 0

    j = 0
    ests = np.zeros((max_iter, g0.shape[0])) + np.nan

    # hacky
    # coef_chunk1 = 0:4
    # coef_chunk2 = 4:104
    # coef_chunk3 = 104:-1

    while diff > tol and iter < max_iter:
        # Set the initial value
        if iter == 0:
            g = g0
        else:
            g = g0 + 0.9 * (g - g0)

        # calc the weight matrix
        W = jnp.diag(calc_weight_diag(X, g))
        # calc the augmented dependent variable
        Z = calc_augmented_depvar(Y, X, g)

        # pdr regression
        g0 = jnp.linalg.solve(X.T @ W @ X, X.T @ W @ Z)
        ests[iter, :] = g0
        diff = jnp.abs(g0 - g).max()

        iter += 1
        j += 1

        # print(g0[0])

    return g0, iter, diff, ests
