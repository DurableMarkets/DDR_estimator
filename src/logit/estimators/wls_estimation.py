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
    # I = ccps.values.flatten() != 0.0

    X = X[model_specification]
    X = X.values.astype(float)

    # X = X[I, :]

    # ccps = ccps.loc[I, :]
    logY = np.log(ccps.values.flatten())


    B = estimate_wls(logY, X, ccps, counts)
    breakpoint()
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

    # jaxlinalgsolve=jax.jit(jnp.linalg.solve)
    # Does nothing to speed up the code.
    # I could consider using vmap instead of a for loop

    g0 = jnp.linalg.solve(xwx, xwy)
    # g0 = jaxlinalgsolve(xwx, xwy)

    return g0


def calculate_weights(ccps,counts):
    # calc weights
    n_ds = counts.fillna(0.0).values

    # if n_ds == 0:
    I = n_ds != 0.0
    # std_ds = np.zeros_like(cfps)
    # std_ds[I] = np.sqrt( cfps[I] * (1 - cfps[I]) / n_ds[I] )

    # In the proces of creating the var_ds some values can get incredibly small and therefore be interpreted as zeros.
    # This is not good since it will lead to division by zero.
    # Therefore I do a new index
    # I = std_ds > 0.0

    # initialize weights
    weights = np.zeros_like(counts)
    weights[I] = np.sqrt(n_ds[I] * ccps[I] / (1 - ccps[I]))
    return weights