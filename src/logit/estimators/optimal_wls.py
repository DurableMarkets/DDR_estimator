import numpy as np
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

    #test_of_covariance_matrices(X, ccps, counts)

    # calc the weights
    weight_blocks = calculate_weights(ccps, counts)
    X_indices = X.index.droplevel([level for level in X.index.names if level not in ["consumer_type", "state"]]).unique()

    #xwx_=np.zeros((114,114))
    # xwx_test = [X.loc[X_indices[i][0],:,X_indices[i][1], :, :].values.T 
    #     @ weight_blocks[i] 
    #     @ X.loc[X_indices[i][0],:,X_indices[i][1], :, :].values 
    #     for i in range(len(weight_blocks))]

    # xwy_test = [X.loc[X_indices[i][0],:,X_indices[i][1], :, :].values.T 
    #     @ weight_blocks[i] 
    #     @ Y.values 
    #     for i in range(len(weight_blocks))]


    # for i in range(len(xwx_test)):
    #     xwx_+=xwx_test[i]

    xw = jnp.concatenate(
        [X.loc[X_indices.get_level_values('consumer_type')[i],:,X_indices.get_level_values('state')[i], :, :].values.T 
        @ weight_blocks[i] for i in range(len(weight_blocks))]
    ,axis=1)

    xwx = xw @ X.values
    xwy = xw @ Y
    
    # if np.allclose(np.linalg.det(xwx),0.0):
    #     print("xwx is singular")

    # This is for testing purposes only
    # x_uwx_u = xwx[0:12,0:12]
    # x_evwx_ev = xwx[12:, 12:]
    # x_uwx_ev = xwx[0:12, 12:]
    # x_evwx_u = xwx[12:, 0:12]

    # # singularity checks
    # np.allclose(x_uwx_ev.T, x_evwx_u) # symmetry check
    # #np.linalg.det(x_uwx_u)
    # #np.linalg.det(x_evwx_ev)
    # #breakpoint()    

    # # WLS regression
    g0 = np.linalg.solve(xwx, xwy)
    #g0= np.linalg.lstsq(xwx, xwy)[0]
    #breakpoint()
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
    if counts.min() == 0:
        raise ValueError("Counts cannot be zero.")

    weight_blocks = []
    for i in range(N.shape[0]):
        consumer_type, state=N.index[i]
        P = ccps.loc[idx[consumer_type, state, :]].values
        K = P.shape[0]
        N_is = N.loc[N.index[i]]

        # A is the pseudo-inverse of B 
        # A= (
        #    np.diag(P)
        #    @
        #    (np.identity(K) - 1/K * np.ones((K,K)))
        #    @ np.diag(1/P) 
        #    @ (np.identity(K) - 1/K * np.ones((K,K))) 
        #    @ np.diag(P)
        # )
        # Pcol=np.c_[P]
        # ocol = np.c_[np.ones(K)]
        # test = 1/K**2 * Pcol @ ocol.T @ np.diag(1/P) @ ocol @ Pcol.T
        # A = np.diag(P) - 1/K * Pcol @ Pcol.T 

        # A= (
        #    (np.identity(K)/K - np.ones((K,K)))
        #    @ np.diag(P) 
        #    @ (np.identity(K)/K - np.ones((K,K))) 
        # )

        
        # similar to diag(P) 
        #A = np.diag(P) - 1/K * np.diag(P) @ np.ones((K,K)) - 1/K * np.ones((K,K)) @ np.diag(P) + 2/(K**2) * np.diag(P) @ np.ones((K,K))


        # THIS one is not correct but does give very pretty errors. 
        # A = np.diag(P) - 1/K * np.diag(P) @ np.ones((K,K)) - 1/K * np.ones((K,K)) @ np.diag(P) + 2/(K**2) * np.diag(P) @ np.ones((K,K))        
        #A = A.T (and not symmetric)
        
        # pesendorfer and Schmidt-Dengler (2008) has this one
        # IT works quite well actually
        # A = np.diag(P) + np.c_[P] @ np.c_[P].T/ P[-1] # P[-1] is the purge decision
        # this one also works nicely - But appararently it is not a generalized inverse... Really strange
        # A = np.diag(P) + np.c_[P] @ np.c_[P].T # just playing with this one to see if it is better
        # This one is a pseudo inverse but it gives us crap....
        # I know why... the xwx matrix becomes singular when we choose this weight matrix.
        # A=(np.diag(P) - np.c_[P] @ np.c_[P].T) 
        # This one is also works fine but it is still not the generalized inverse. 
        #A=(np.diag(P) - np.c_[P] @ np.c_[P].T/P[-1]) # P[-1] is the purge decision


        # Playing around: 
        # A is the pseudo-inverse of B 
        # A= (
        #    np.diag(P)
        #    @
        #    (np.identity(K) - 1/K * np.ones((K,K)))
        #    @ np.diag(1/P) 
        #    @ (np.identity(K) - 1/K * np.ones((K,K))) 
        #    @ np.diag(P)
        # )
        # A=(np.diag(P) - np.c_[P] @ np.c_[P].T) 
        #A = np.diag(1/P) - np.ones((K,K))
        #A = np.diag(1/P) + np.ones((K,K))
        #A = np.diag(P) - np.c_[P] @ np.c_[P].T/ P[-1] # P[-1] is the purge decision

        #A=(np.diag(P) - 1/K *np.c_[P] @ np.c_[P].T) 
        # A = (
        #     (np.identity(K) + 1/np.sum(P**2) *  np.c_[P] @ np.c_[P].T )
        #     @ np.diag(P)
        #     @ (np.identity(K) + 1/np.sum(P**2) *  np.c_[P] @ np.c_[P].T )
        # )
             #np.diag(P) - 1/K * np.c_[P] @ np.c_[P].T # This one is not a generalized inverse but it works well.
        #A = np.diag(P**2) # This one because it would equivalent to minimizing the squared conjugate social surplus. 
        # A =(    
        #     np.diag(P)
        #     - 1/np.sum(P**2) * np.c_[P] @ np.c_[P].T @ np.diag(P)  
        #     - 1/np.sum(P**2) * np.diag(P) @ np.c_[P] @ np.c_[P].T  
        #     - np.sum(P**3)/(np.sum(P**2)**2) * np.c_[P] @ np.c_[P].T 
        # )
        # This one works well, (not better than P). I havent checked if its a generalized inverse, but is derived based on the matrix cookbook. 
        # It is a generalized inverse of P - ee.T but unfortunately not the other way around.
        # I think there was a sign error so we should get. 
        # these two seem to be similar...
        
        # A =(
        #     np.diag(P)
        #     - 1/np.sum(P**2) * np.c_[P] @ np.c_[P].T @ np.diag(P)  
        #     - 1/np.sum(P**2) * np.diag(P) @ np.c_[P] @ np.c_[P].T  
        #     + np.sum(P**3)/(np.sum(P**2)**2) * np.c_[P] @ np.c_[P].T 
        # )

        # This one is a reflexive generalized inverse of P - ee.T.
        # A =(
        # (np.identity(K) - 1/np.sum(P**2) * np.c_[P] @ np.c_[P].T) 
        # @ np.diag(P)
        # @(np.identity(K) - 1/np.sum(P**2) * np.c_[P] @ np.c_[P].T) 
        # )
        # but seems to be unstable

        # This works well...
        # A =(
        # (np.identity(K) -  np.c_[P] @ np.c_[P].T) 
        # @ np.diag(P)
        # @(np.identity(K) - np.c_[P] @ np.c_[P].T) 
        # )

        A = ( 
        (np.identity(K) -  1/np.sum(P**2) *np.c_[P] @ np.c_[P].T) 
        @ np.diag(P)
        @(np.identity(K) - 1/np.sum(P**2) *np.c_[P] @ np.c_[P].T) 
        )

        #A = np.diag(P) - np.c_[P] @ np.c_[P].T
        #A[-1,-1] = 1

        #breakpoint()
        A = ((np.identity(K) - 1/K * np.ones((K,K)))
            @ np.diag(1/P) 
            @ (np.identity(K) - 1/K * np.ones((K,K)))
        )



        #ABA = A @ B @ A 


        # #if check_if_reflexive_generalized_inverse(A, np.diag(1/P) - np.ones((K,K))) == False: 
        #     print("Not reflexive generalized inverse")
        # if check_if_symmetric(A, np.diag(1/P) - np.ones((K,K))) == True:
        #     print("Not symmetric")


        # It is the last_term that just messes everything up. I think it must be numerically very unstable or something
    
        #second_term = - 1/K * (np.c_[P] @ np.c_[np.ones(K)].T +  np.c_[np.ones(K)] @ np.c_[P].T) 
        #last_term = 1/(K**2) * np.c_[P] @ np.c_[np.ones(K)].T @ np.diag(1/P) @ np.c_[np.ones(K)] @ np.c_[P].T
        
        #A = np.diag(P) +  second_term + last_term
        #breakpoint()
        #np.diag(P) @ np.ones((K,K)) @ np.diag(1/P) @ np.ones((K,K)) @ np.diag(P)
        #breakpoint()
        #A = np.identity(K)
        #breakpoint()

        #breakpoint()
        #A = (np.diag(P) - 1/K * np.c_[P] @ np.c_[np.ones(K)].T - 1/K * np.c_[np.ones(K)] @ np.c_[P].T) # this works well
        #A = np.diag(P) - 1/K * np.ones((K,K))
        #A= np.diag(P) # This just works the best...

        #A = np.identity(K)
        #A = np.diag(P) - np.c_[P] @ np.c_[P].T
        # B = np.diag(1/P) - np.ones((K,K))
        # #A = np.linalg.pinv(B) 
        # ABA=A @ B @ A 
        # #breakpoint()
        # Symmetry check
        # np.allclose(A.T, A)

        # np.allclose(A, ABA)

        # type 1
         

        # type 2

        #A = N_is/N_all*np.diag(P) - np.c_[P] * np.c_[P].T # This works the best so far but is technically wrong...
        #A = N_is/N_all*(np.diag(P) - np.c_[P] * np.c_[P].T) 
        #A = (np.diag(P) - np.c_[P] * np.c_[P].T)

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
        #A = np.diag(1/P) - np.ones((K,K))

        #A = np.identity(K)
        weight_blocks.append(A)

    return weight_blocks

def check_if_reflexive_generalized_inverse(A,B):
    ABA=A @ B @ A 
    BAB=B @ A @ B
    return np.all([np.allclose(A, ABA), np.allclose(BAB, B)])
def check_if_symmetric(A,B):
    AB=A @ B 
    BA=B @ A 
    return np.all([np.allclose(AB,AB.T), np.allclose( BA, BA.T)])

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
    A = np.diag(P) - np.c_[P] @ np.c_[P].T/ P[-1] # P[-1] is the purge decision
    A = np.diag(P) - np.c_[P] @ np.c_[P].T # P[-1] is the purge decision
    B = np.diag(1/P)
    
    B = np.diag(1/P) - np.ones((K,K))
    A =(
        np.diag(P)
        - 1/np.sum(P**2) * np.c_[P] @ np.c_[P].T @ np.diag(P)  
        - 1/np.sum(P**2) * np.diag(P) @ np.c_[P] @ np.c_[P].T  
        - np.sum(P**3)/(np.sum(P**2)**2) * np.c_[P] @ np.c_[P].T 
    )

    A =(
    (np.identity(K) - 1/np.sum(P**2) * np.c_[P] @ np.c_[P].T) 
    @ np.diag(P)
    @(np.identity(K) - 1/np.sum(P**2) * np.c_[P] @ np.c_[P].T) 
    )


    ABA=A @ B @ A 
    BAB=B @ A @ B

    print(np.allclose(A, ABA))
    print(np.allclose(BAB, B))

    return None



def test_of_covariance_matrices(X, ccps, counts):

    X = X.astype(float)

    # diag(P)
    X_indices = X.index.droplevel([level for level in X.index.names if level not in ["consumer_type", "state"]]).unique()

    weights=calculate_weights(ccps, counts)

    xw = jnp.concatenate(
        [(X.loc[X_indices[i][0],:,X_indices[i][1], :, :].values.T 
        @ weights[i]) for i in range(len(weights))]
    ,axis=1)
    xwx = xw @ X.values

    xdiagp = jnp.concatenate(
        [(X.loc[X_indices[i][0],:,X_indices[i][1], :, :].values.T 
        @ np.diag(ccps.loc[X_indices[i][0], X_indices[i][1], :])).astype(float) for i in range(len(X_indices))]
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

    # are they generalized inverses? 
    A = np.linalg.inv(xwx @ xwx) @ xwx
    B = xdiagpx - xppx
    cand = np.linalg.inv(xdiagpx) @ (xdiagpx - xppx) @ np.linalg.inv(xdiagpx)
    B_sword = np.linalg.pinv(B)
    np.allclose(B_sword, xdiagpx)

    ABA = A @ B @ A
    BAB = B @ A @ B

    np.allclose(A, ABA)
    np.allclose(B, BAB)


    return weights
