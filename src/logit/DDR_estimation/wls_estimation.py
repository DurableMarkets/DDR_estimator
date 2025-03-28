import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def calculate_weights(counts):
    # calc frequencies:
    counts.loc[:, "cfps"] = counts["counts"] / counts.groupby(["tau", "state"])[
        "counts"
    ].transform("sum")

    # calc weights
    cfps = counts["cfps"].values
    n_ds = counts["counts"].fillna(0.0).values

    # if n_ds == 0:
    I = n_ds != 0.0
    # std_ds = np.zeros_like(cfps)
    # std_ds[I] = np.sqrt( cfps[I] * (1 - cfps[I]) / n_ds[I] )

    # In the proces of creating the var_ds some values can get incredibly small and therefore be interpreted as zeros.
    # This is not good since it will lead to division by zero.
    # Therefore I do a new index
    # I = std_ds > 0.0

    # initialize weights
    weights = np.zeros_like(cfps)
    # weights[I] = np.sqrt(n_ds[I]/ (cfps[I] * (1 - cfps[I])))
    weights[I] = np.sqrt(n_ds[I] * cfps[I] / (1 - cfps[I]))
    return weights


# @profile
def estimate_wls(Y, X, counts):
    """Estimates the WLS regression."""
    # Pad Y nans with zeros
    Y = jnp.nan_to_num(Y, nan=0.0)

    # calc the weights
    weights = calculate_weights(counts)

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
