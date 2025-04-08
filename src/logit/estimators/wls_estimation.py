import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

jax.config.update("jax_enable_x64", True)

def wls_regression_mc(X, ccps, counts, model_specification):
    """This function estimates the parameters of the DDR regression.

    ccp can be any ccp estimator as long as the shape is consistent with
    state_decision_arrays

    """
    # Index for zero share rows

    X = X[model_specification]
    X = X.values.astype(float)

    logY = np.log(ccps.values.flatten())

    B = estimate_wls(logY, X, ccps, counts)
    est = pd.DataFrame(B, index=[model_specification], columns=["Coefficient"])

    return est

# @profile
def estimate_wls(Y, X, ccps, counts):
    """Estimates the WLS regression."""
    # Pad Y nans with zeros
    Y = jnp.nan_to_num(Y, nan=0.0)

    # calc the weights
    weights = calculate_weights(ccps, counts)

    # calc weighted data
    xw = jnp.multiply(X.T, weights)
    xwx = xw @ X
    xwy = xw @ Y

    # WLS regression

    g0 = jnp.linalg.solve(xwx, xwy)

    return g0


def calculate_weights(ccps,counts):
    # calc weights
    n_ds = counts.fillna(0.0).values

    # if n_ds == 0:
    I = n_ds != 0.0

    # initialize weights
    weights = np.zeros_like(counts)
    weights[I] = np.sqrt(n_ds[I] * ccps[I] / (1 - ccps[I]))
    return weights