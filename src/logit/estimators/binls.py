import numpy as np
from pandas import IndexSlice as idx
import jax.numpy as jnp
import pandas as pd

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
    
    #X = X.values.astype(float)

    logY = np.log(ccps.values.flatten())

    B, est_post = estimate_owls(logY, X, ccps, counts)
    est = pd.DataFrame(B, index=[model_specification], columns=["Coefficient"])

    return est, est_post


def estimate_owls(Y, X, ccps, counts):
    """This function estimates the parameters of the DDR regression using the optimal wls weight matrix. 

    """
    # Index for zero share rows
    Y = jnp.nan_to_num(Y, nan=0.0)
    X = X.astype(float)
    
    # calc the weights
    weight_blocks = calculate_weights(ccps, counts)
    X_indices = X.index.droplevel([level for level in X.index.names if level not in ["consumer_type", "state"]]).unique()
    #breakpoint()
    xw = jnp.concatenate(
        [X.loc[X_indices[i][0],:,X_indices[i][1], :, :].values.T 
        @ weight_blocks[i] for i in range(len(weight_blocks))]
    ,axis=1)
    xwx = xw @ X.values
    xwy = xw @ Y

    # WLS regression

    g0 = np.linalg.solve(xwx, xwy)

    preds = X.values @ g0
    residuals = Y - preds
    est_post=pd.DataFrame(
        data={'preds':preds,
              'residuals': 
              residuals, 
              'Y': Y,
              'ccps': ccps.values, 
              'counts': counts.values,
              },
              index=X.index
    )

    return g0, est_post
    


def calculate_weights(ccps, counts):
    

    # initialize 
    N = counts.groupby(
        ["consumer_type", "state"]
    ).sum()
    N_all = counts.sum()

    weight_blocks = []
    for i in range(N.shape[0]):
        consumer_type, state=N.index[i]
        P = ccps.loc[idx[consumer_type, state, :]].values
        K = P.shape[0]
        N_is = N.loc[N.index[i]]
        #breakpoint()
        #A = N_is/ N_all * np.diag(P - P**2) # This works better for some reason...
        #A = np.diag(1/(P - P**2))
        A = np.diag(P - P**2)
        weight_blocks.append(A)

    return weight_blocks
