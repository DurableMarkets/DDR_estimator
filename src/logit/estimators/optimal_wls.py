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
    breakpoint()
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

    return g0
    


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
        # A is the pseudo-inverse of B 
        A= N_is/N_all * (
           np.diag(P)
           @
           (np.identity(K) - 1/K * np.ones((K,K)))
           @ np.diag(1/P) 
           @ (np.identity(K) - 1/K * np.ones((K,K))) 
           @ np.diag(P)
        )

        # Test 1
        #A= N_is/N_all * (np.identity(K))
        #A=  (np.identity(K))

        # A= (
        #     np.diag(P)
        #     @(np.identity(K) - np.ones((K,K)))
        #     @ np.diag(1/P) 
        #     @ (np.identity(K) - 1/K * np.ones((K,K))) 
        #     @ np.diag(P)
        # )
        # Test 2 
        # Try N_is/N diag(p_sd - p_sd**2)

        #A = 1/N_is *(np.diag(P) - np.c_[P] * np.r_[P])
        #A = np.diag(P) - np.c_[P] * np.r_[P]

        #A = np.identity(K)
        weight_blocks.append(A)

    return weight_blocks

