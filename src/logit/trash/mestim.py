import time

import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize
from tabulate import tabulate


# Some estimation routines
def estimation(
    Qfun, theta0, deriv=0, cov_type="sandwich", parnames="", counts=None, output=False
):
    """
    estimation: Generic routine for M-estimation

    args:
        - Qfun: function that returns objective function and derivatives
        - theta0: K x 1 vector of initial values for parameters
        - deriv: 0: use numerical derivatives, 1: use user-supplied derivatives, 2: use user-supplied Hessian
        - cov_type: type of variance covariance matrix, can be 'Ainv', 'Binv', 'sandwich'
        - parnames: list of parameter names
        - output: if True, print output

    returns:
        - res: dictionary with results, with fields
            - theta_hat: K x 1 vector of parameter estimates
            - se: K x 1 vector of standard errors
            - t-values: K x 1 vector of t-values
            - cov: K x K variance covariance matrix
            - Q: value of objective function at optimum
            - time: elapsed time
            - s_i: N x K array of scores
    """

    tic = time.perf_counter()

    # Q: Sample objective function to minimize (e.g. sample average of negative log-likelihood)
    Q = lambda theta: Qfun(theta, out="Q")

    # dQ: Derivative of sample objective function wrt parameters theta (function returns size K array)
    dQ = None
    if deriv > 0:  # use user-supplied 1 order derivatives
        dQ = lambda theta: Qfun(theta, out="dQ")

    # Define your callback function
    def print_iteration(nit):
        print(".")
        # print(f"Parameter vector: {x}")

    hess = None
    if deriv > 1:  # use user-supplied 2 order derivatives
        hess = lambda theta: Qfun(theta, out="H")
        res = minimize(
            fun=Q,
            jac=dQ,
            x0=theta0,
            hess=hess,
            method="trust-exact",
            callback=print_iteration,
        )
        res.hess_inv = la.inv(res.hess)
    else:  # use bfgs
        res = minimize(
            fun=Q, jac=dQ, x0=theta0, method="bfgs", callback=print_iteration
        )

    theta_hat = np.array(res.x).reshape(-1, 1)

    toc = time.perf_counter()

    # variance co-variance matrix
    s_i = Qfun(theta_hat, out="s_i")

    cov = avar(s_i, res.hess_inv, counts, cov_type)
    se = np.sqrt(np.diag(cov)).reshape(-1, 1)

    # collect output
    names = ["parnames", "theta_hat", "se", "t-values", "cov", "Q", "time", "s_i"]
    results = [
        parnames,
        theta_hat,
        se,
        theta_hat / se,
        cov,
        Q(theta_hat),
        toc - tic,
        s_i,
    ]

    res.update(dict(zip(names, results)))

    if output:
        if res.parnames:
            table = {
                k: res[k] for k in ["parnames", "theta_hat", "se", "t-values", "jac"]
            }
        else:
            table = {k: res[k] for k in ["theta_hat", "se", "t-values", "jac"]}
        print(tabulate(table, headers="keys", floatfmt="10.5f"))
        print("")
        print(res.message)
        print("Objective function:", res["Q"])
        print(
            "Iteration info: %d iterations, %d evaluations of objective, and %d evaluations of gradients"
            % (res.nit, res.nfev, res.njev)
        )
        print(f"Elapsed time: {res['time']:0.4f} seconds")

    return res


def avar(s_i, Ainv, counts=None, cov_type="sandwich"):
    if counts is None:
        n, K = s_i.shape
    else:
        n = counts.sum()
    B = s_i.T @ s_i / n
    if cov_type == "Ainv":
        return Ainv / n
    if cov_type == "Binv":
        return la.inv(B) / n
    if cov_type == "sandwich":
        return Ainv @ B @ Ainv / n
