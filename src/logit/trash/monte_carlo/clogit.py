import mestim as m
import numpy as np
from tabulate import tabulate


# Softrware to estimate conditional logit model
def clogit(
    y, x, counts=None, cov_type="Ainv", theta0=None, deriv=0, parnames=None, quiet=False
):
    """
    clogit: Estimate conditional logit model

    args:
        - y: N x 1 vector of choices
        - x: N x J x K array of choice specific variables
        - cov_type: type of variance covariance matrix, can be 'Ainv', 'Binv', 'sandwich'
        - theta0: K x 1 vector of initial values for parameters
        - deriv: 0: use numerical derivatives, 1: use user-supplied derivatives, 2: use user-supplied Hessian
        - quiet: if False, print output

    returns:
        - res: dictionary with results, with fields
            - theta_hat: K x 1 vector of parameter estimates
            - se: K x 1 vector of standard errors
            - t-values: K x 1 vector of t-values
            - cov: K x K variance covariance matrix

    """
    # Objective function and derivatives for conditional logit model
    N, J, K, palt, xalt, xvars = labels(x)

    if counts is None:
        counts = np.ones(N)

    if parnames is None:
        parnames = xvars

    Qfun = lambda theta, out: Q_clogit(theta, y, x, counts, out)

    if theta0 is None:
        theta0 = np.zeros(K)

    res = m.estimation(Qfun, theta0, deriv, cov_type, parnames, counts=counts)
    res.update(
        dict(
            zip(
                ["yvar", "xvars", "K", "N_cells", "N_obs"],
                ["y", xvars, K, N, np.sum(counts)],
            )
        )
    )

    if quiet == False:
        print("Conditional logit")
        print("Initial log-likelihood", -Qfun(theta0, "Q"))
        print("Initial gradient\n", -Qfun(theta0, "dQ"))
        print_output(res)

    return res


def Q_clogit(theta, y, x, counts, out="Q"):
    """
    Q_clogit: Objective function and derivatives for conditional logit model

    args:
        - theta: K x 1 vector of parameters
        - y: N x 1 vector of choices
        - x: N x J x K array of choice specific variables
        - counts: N x 1 vector of counts (ones if micro data, counts if aggregate data)
        - out: 'Q', 'dQ', 'H', 'predict', 's_i'

    returns:
        - Q: value of objective function at optimum
        - g: K x 1 vector of gradients
        - H: K x K Hessian matrix
        - v: N x J array of deterministic utilities
        - p: N x J array of choice probabilities
        - dv: N x J x K array of choice specific derivatives of utility wrt parameters
        - s_i: N x K array of scores

    """

    counts_i = counts.reshape(-1, 1)
    v = utility(theta, x)  # Deterministic component of utility
    ll_i = logccp(v, y)
    q_i = -ll_i

    if out == "Q":
        return np.sum(q_i * counts_i) / np.sum(counts_i)
    #        return np.mean(q_i)

    dv = x
    p = ccp(v)
    if out == "predict":
        return v, p, dv  # Return predicted values
    N, J, K = dv.shape
    idx = y[:,] + J * np.arange(0, N)
    dvj = dv.reshape(N * J, K)[idx, :]  # pick choice specific values corresponding to y

    s_i = dvj - np.sum(p.reshape(N, J, 1) * dv, axis=1)
    # g = -np.mean(s_i, axis=0)
    g = -np.sum(s_i * counts_i, axis=0) / np.sum(counts_i)

    s_i = s_i * np.sqrt(np.max(counts_i, 1)).reshape(-1, 1)

    if out == "s_i":
        return s_i  # Return s_i: NxK array with scores
    if out == "dQ":
        return g
        # Return dQ: array of size K derivative of sample objective function
    if out == "H":
        return (
            s_i.T @ s_i / np.sum(counts_i)
        )  # Return jac: 1xK array with derivative of sample objective function


def utility(theta, x):
    N, J, K = x.shape
    u = x @ theta
    return u.reshape(N, J)


def logsum(v, sigma=1):
    """
    logsum:
        Expected max over iid extreme value shocks with scale parameter sigma
            Logsum is reentered around maximum to obtain numerical stability (avoids overflow, but accepts underflow)

    Args:
        - v: N x J array of values
        - sigma: scale parameter

    Returns:
        - logsum: N x 1 array of logsums

    """
    v = np.array(v)
    max_v = v.max(axis=1).reshape(-1, 1)
    return max_v + sigma * np.log(np.sum(np.exp((v - max_v) / sigma), 1)).reshape(-1, 1)


def logccp(v, y=None, sigma=1):
    # Log of conditional choice probabilities
    # If y=None return logccp corresponding to all choices
    # if y is Nx1 vector of choice indexes, return likelihood

    ev = logsum(v, sigma)  # Expected utility (always larger than V)
    if y is not None:
        N, J = v.shape
        idx = y[:,] + J * np.arange(0, N)
        v = v.reshape(N * J, 1)[idx]  # pick choice specific values corresponding to y
    return (v - ev) / sigma


def ccp(v, sigma=1):
    ev = logsum(v, sigma)  # Expected utility (always larger than V)

    return np.exp(v / sigma) / np.exp(ev / sigma)


def logit_ccp(v, y=None, sigma=1):
    # Conditional choice probabilities
    return np.exp(logccp(v, y, sigma))


def labels(x):
    # labels and dimensions for plotting
    N, J, K = x.shape
    palt = ["p" + str(i) for i in range(J)]
    xalt = ["alt" + str(i) for i in range(J)]
    xvars = ["var" + str(i) for i in range(K)]
    return N, J, K, palt, xalt, xvars


def print_output(res, cols=["parnames", "theta_hat", "se", "t-values", "jac"]):
    print("Dep. var. :", res["yvar"], "\n")

    table = {k: res[k] for k in cols}
    print(tabulate(table, headers="keys", floatfmt="10.5f"))
    # print('\n# of groups:      :', res['n'])
    print("# of observations :", res["N_obs"])
    print("# of cells :", res["N_cells"])
    print("log-likelihood. :", -res["Q"] * res["N_obs"], "\n")
    print("average log-likelihood. :", -res["Q"], "\n")
    print(
        "Iteration info: %d iterations, %d evaluations of objective, and %d evaluations of gradients"
        % (res.nit, res.nfev, res.njev)
    )
    print(f"Elapsed time: {res['time']:0.4f} seconds")
    print("")
