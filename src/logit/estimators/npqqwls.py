import numpy as np
from pandas import IndexSlice as idx
import jax.numpy as jnp
import pandas as pd
import jax
from itertools import compress
jax.config.update("jax_enable_x64", True)

def owls_regression_mc(X, ccps, counts, model_specification):
    """This function estimates the parameters of the DDR regression.

    ccp can be any ccp estimator as long as the shape is consistent with
    state_decision_arrays

    """
    # Index for zero share rows
    I=(counts != 0).values

    ccps = ccps.loc[I] # removes all 0 counts
    counts = counts.loc[I] # removes all 0 counts

    X = X[model_specification].loc[I]

    logY = np.log(ccps.values.flatten())

    B, se, est_post, rank_deficiency_mask = estimate_owls(logY, X, ccps, counts)

    # remove rank deficient varnames
    model_specification = list(compress(model_specification, rank_deficiency_mask))
    if se is not None:
        est = pd.DataFrame(np.array([B,se]).T, index=[model_specification], columns=["Coefficient", 'se'])
    else:
        est = pd.DataFrame(B, index=[model_specification], columns=["Coefficient"])

    return est, est_post


def estimate_owls(Y, X, ccps, counts):
    """This function estimates the parameters of the DDR regression using the optimal wls weight matrix. 

    """
    # Index for zero share rows
    Y = np.nan_to_num(Y, nan=0.0)
    X = X.astype(float)

    # calc the weights
    weight_blocks = calculate_weights(ccps, counts)
    X_indices = X.index.droplevel([level for level in X.index.names if level not in ["consumer_type", "state"]]).unique()

    xw = np.concatenate(
        [X.loc[X_indices.get_level_values('consumer_type')[i],:,X_indices.get_level_values('state')[i], :, :].values.T 
        @ weight_blocks[i] for i in range(len(weight_blocks))]
    ,axis=1)

    xwx = xw @ X.values
    xwy = xw @ Y

    # Remove any variables that evaluate to zero 
    rank_deficiency_mask = xwx.sum(axis=1) != 0.0
    xwx=xwx[np.ix_(rank_deficiency_mask, rank_deficiency_mask)]
    xwy=xwy[rank_deficiency_mask]

    # WLS regression
    g0 = np.linalg.solve(xwx, xwy)

    # Compute Avar
    avar=calculate_asymptotic_var(ccps, counts, X, xwx)
    se=np.sqrt(np.diag(avar)/counts.sum())

    # Some extra diagnostics
    preds = X.values[:, rank_deficiency_mask] @ g0
    residuals = Y - preds
    est_post=pd.DataFrame(
        data={'preds':preds,
              'residuals': 
              residuals, 
              'Y': Y,
              'ccps': ccps.values, 
              'counts': counts.values},
              index=X.index
    )

    return g0, se, est_post, rank_deficiency_mask
    
def calculate_weights(ccps, counts):
    
    # initialize 
    N = counts.groupby(
        ["consumer_type", "state"]
    ).sum()
    N_all = counts.sum()
    if counts.min() == 0:
        raise ValueError("Counts cannot be zero.")

    weight_blocks = []
    for i in range(N.shape[0]):
        consumer_type, state=N.index[i]
        P = ccps.loc[idx[consumer_type, state, :]].values
        K = P.shape[0]
        N_is = N.loc[N.index[i]]

        A = N_is/N_all*(np.diag(P) + np.c_[P] @ np.c_[P].T)
        weight_blocks.append(A)

    return weight_blocks

def calculate_asymptotic_var(ccps, counts, X, xwx): 
        # # removing any rank deficiencies   
        xwx_inv = np.linalg.inv(xwx)
        return xwx_inv     
