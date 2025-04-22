import numpy as np
from pandas import IndexSlice as idx
import jax.numpy as jnp
import pandas as pd

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
 
    #A=(np.identity(K) - 1/K * np.ones((K,K)))@ np.diag(1/P) @ (np.identity(K) - 1/K * np.ones((K,K)))
    # Testing now if my hypothesis is correct
    A= 1/N * (
        np.diag(P)
        @(np.identity(K) - np.ones((K,K)))
        @ np.diag(1/P) 
        @ (np.identity(K) - 1/K * np.ones((K,K))) 
        @ np.diag(P)
    )
    #A = np.diag(P) - np.c_[P] * np.r_[P]
    
    B = N * (np.diag(1/P) - np.ones((K,K)))   

    ABA=A @ B @ A 

    np.allclose(A, ABA)

    BAB=B @ A @ B 

    np.allclose(B, BAB)

    return weights

def owls_regression_mc(X, ccps, counts, model_specification):
    """This function estimates the parameters of the DDR regression.

    ccp can be any ccp estimator as long as the shape is consistent with
    state_decision_arrays

    """
    # Index for zero share rows

    X = X[model_specification]
    #X = X.values.astype(float)

    logY = np.log(ccps.values.flatten())

    B = estimate_owls(logY, X, ccps, counts)
    est = pd.DataFrame(B, index=[model_specification], columns=["Coefficient"])

    return est


def estimate_owls(Y, X, ccps, counts):
    """This function estimates the parameters of the DDR regression using the optimal wls weight matrix. 

    """
    # Index for zero share rows
    Y = jnp.nan_to_num(Y, nan=0.0)
    X = X.astype(float)
    
    # calc the weights
    XwX_blocks, XwY_blocks = calculate_blocks(ccps, X, counts)

    # TODO: there is a bug here. The shape is off. 
    xwx = jnp.concatenate(XwX_blocks, axis=0)
    xwy = jnp.concatenate(XwY_blocks, axis=0)
    
    # WLS regression
    #breakpoint()
    g0 = np.linalg.solve(xwx, xwy)

    return g0
    


def calculate_blocks(ccps, X, counts):
    
    # initialize 
    N = counts.groupby(
        ["consumer_type", "state"]
    ).sum()

    N_all = counts.sum()

    XwX_blocks = []
    XwY_blocks = []
    for i in range(N.shape[0]):
        consumer_type, state=N.index[i]
        P = ccps.loc[idx[consumer_type, state, :]].values
        K = P.shape[0]
        N_is = N.loc[N.index[i]]

        X_is=X.loc[consumer_type,:,state, :, :].astype(float).values
        
        Xq=X_is.T @ P

        XqX = X_is.T @ np.diag(P) @ X_is

        XwX = XqX - Xq @ Xq.T

        #breakpoint()
        XwY = (X_is.T @ np.diag(P) @ np.log(P)  - X_is.T @ np.c_[P] @ np.c_[P].T @ np.log(P))

        XwX_blocks.append(N_is/N_all*XwX)
        XwY_blocks.append(N_is/N_all*XwY)

    return XwX_blocks, XwY_blocks

