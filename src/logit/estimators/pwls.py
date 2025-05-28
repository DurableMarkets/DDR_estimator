import numpy as np
from numpy.linalg import inv
from pandas import IndexSlice as idx
import jax.numpy as jnp
import pandas as pd
import jax
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

    xw = jnp.concatenate(
        [X.loc[X_indices.get_level_values('consumer_type')[i],:,X_indices.get_level_values('state')[i], :, :].values.T 
        @ weight_blocks[i] for i in range(len(weight_blocks))]
    ,axis=1)

    xwx = xw @ X.values
    xwy = xw @ Y

    # WLS regression
    g0 = np.linalg.solve(xwx, xwy)

    # wls Avar 
    Avar = Avar(xwx, xw, PsigmaP)


    preds = X.values @ g0
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

    return g0, est_post
    
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

        # A is the pseudo-inverse of B 
        A= np.diag(P)
        #A = np.diag(N_is/N_all*P)
        #A= N_is/ N_all * np.diag(P) # essentially weighing by n_isd and scaling by N_all
        weight_blocks.append(A)

    return weight_blocks

def Avar(xw, xwx, P):
    Sigma = np.diag(1/P) - np.ones((P.shape[0],P.shape[0]))
    
    return inv(xwx)@ xw @ Sigma @ xw.T @ inv(xwx) 

def playground_test_of_pseudo_inverses(ccps, counts):
    # calc weights
    # counts
    N_state = counts.groupby(
        ["consumer_type", "state"]
    ).sum()

    # diag(ccps) - np.ones @ np.ones.T
    # Try with one state to see if it is invertible
    Pinvdiag=np.diag(1/ccps.loc[0,0,:].values)
    W_inner = (1/N_state.loc[0,0])* (Pinvdiag - np.ones_like(Pinvdiag))
    # Incredibly close to being singular

    
    # I need to come up with an analytical expression for the pseudoinverse. 
    P = ccps.loc[0,0,:].values
    K = P.shape[0]

    N = N_state.loc[0,0]
    #breakpoint()
    #A=(np.identity(K) - 1/K * np.ones((K,K)))@ np.diag(1/P) @ (np.identity(K) - 1/K * np.ones((K,K)))
    # Testing now if my hypothesis is correct
    # diag^-1(P) should also be a generalized inverse of diag(P)- P @ P.T
    # That would mean that
    # diag^-1(P) @ (diag(P)- P @ P.T) @ diag^-1(P) = diag^-1(P) - ee.T 
    # has the generalized inverse
    # diag(P) 
    # test this. this is true, but what I need as weight matrix is something that diag^-1(P) - ee.T is a generalized inverse for. 
    
    
    A= N * (
        np.diag(P)
        @(np.identity(K) - 1/K *np.ones((K,K)))
        @ np.diag(1/P) 
        @ (np.identity(K) - 1/K * np.ones((K,K))) 
        @ np.diag(P)
    )
    A = np.diag(P) - np.c_[P] * np.r_[P]
    A = np.diag(P)
    B = N * (np.diag(1/P) - np.ones((K,K)))   
    B = 1/N *(np.diag(1/P) - np.ones((K,K)))
    ABA=A @ B @ A 

    # Symmetry check
    np.allclose(A.T, A)

    np.allclose(A, ABA)

    BAB=B @ A @ B 

    np.allclose(B, BAB)

    return weights



def test_of_covariance_matrices(X, ccps, counts):

    P = ccps.loc[0,0,:].values
    K = P.shape[0]
    #X = X.loc[0,:,0,:,:].values

    # diag(P)
    X_indices = X.index.droplevel([level for level in X.index.names if level not in ["consumer_type", "state"]]).unique()
    xdiagp = jnp.concatenate(
        [X.loc[X_indices[i][0],:,X_indices[i][1], :, :].values.T 
        @ np.diag(ccps.loc[X_indices[i][0], X_indices[i][1], :]) for i in range(len(X_indices))]
    ,axis=1)
   # breakpoint()
    xdiagpx = xdiagp @ X.values
    
    xpp = jnp.concatenate(
        [X.loc[X_indices[i][0],:,X_indices[i][1], :, :].values.T 
        @ np.c_[ccps.loc[X_indices[i][0], X_indices[i][1], :]] 
        @ np.c_[ccps.loc[X_indices[i][0], X_indices[i][1], :]].T for i in range(len(X_indices))]
        ,axis=1)

    xppx = xpp @ X.values
    #breakpoint()
    covdiagP=np.linalg.inv(xdiagpx) - np.linalg.inv(xdiagpx) @ xppx @ np.linalg.inv(xdiagpx)
    # diag

    covpinv=np.linalg.inv(xdiagpx - xppx)
    # both are singular... I think this cannot be done, since we are only considering one state
    return weights
