# Model interface for the EQB model
#
# Convenient helper functions for interacting with the model solution/output/data
# coming out of the jax code.
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, "../../src/")
sys.path.insert(0, "tests/resources/simple_model/")

import jax.numpy as jnp


def logsum(v, sigma):
    vmax = v.max(axis=1, keepdims=True)
    ev = np.exp((v - vmax) / sigma)
    sum_ev = ev.sum(axis=1)
    return sigma * np.log(sum_ev) + v.max(axis=1)  # vmax without keepdims


def get_scrap_prob(
    tau: int, state_decision_arrays: dict, params: dict, prices: dict, options
) -> float:
    """Computes all scrap probabilities for a single car *type*, i.e. for all car ages
    1, ..., abar."""
    state_space = state_decision_arrays["state_space"]
    ncar_types = options["n_car_types"]
    car_types = np.arange(1, ncar_types + 1)

    # car_type_idx = car_type - 1 # car types are 1,2,3,..., but we need 0,1,2,...

    if "mum" in params.keys():
        mum = params["mum"][tau]
        sigma = params["sigma_sell_scrapp"]
        mum2sigma = mum / sigma
    else:
        mum2sigma = params["mum2sigma"]

    # create matrix with all used car prices
    assert (
        np.unique(options["max_age_of_car_types"]).size == 1
    ), "All car types must have the same maximum age for this code to work"
    abar = options["max_age_of_car_types"][0]
    prices_u_all = np.empty((ncar_types, abar - 1))
    for i, car_type in enumerate(car_types):
        all_idx = state_decision_arrays["map_state_to_price_index"][
            car_type
        ]  # indices for prices_used
        max_age_car_type = options["max_age_of_car_types"][
            car_type - 1
        ]  # max age of particular car type
        idx = all_idx[
            1:max_age_car_type
        ]  # remove 0 (new car) and abar (clunker) which cannot be bought
        prices_u_all[i, :] = prices["used_car_prices"][
            idx
        ]  # prices for car type 1,2,3,...,abar

    # Selecting used car prices for the given car_type
    prob_scrap = np.nan * np.empty((ncar_types, abar))
    for car_type_idx in np.arange(ncar_types):
        car_type = car_type_idx + 1  # I get 0,1,... but I need 1,2,...
        assert car_type in car_types, f"car_type={car_type} is not in {car_types}"

        pscrap = get_price_scrap(car_type, state_decision_arrays, params)

        prices_u = prices_u_all[car_type_idx, :]
        n = prices_u.size
        v = np.zeros((n, 2))
        v[:, 0] = mum2sigma * (pscrap - prices_u)
        ev = np.exp(v - v.max(axis=1, keepdims=True))
        prob_scrap_car_type = ev[:, 0] / ev.sum(axis=1)

        prob_scrap[car_type_idx, :-1] = prob_scrap_car_type
        prob_scrap[car_type_idx, -1] = 1.0  # Pr(scrap|abar) = 100%

    return prob_scrap


def get_scrapvalue(
    car_type,
    car_age,
    params,
    options,
    price_selling,
    consumer_type,
):
    """Computes the expected value and the CCP of scraping vehicles versus selling."""
    clunker_age_of_car_type = options["max_age_of_car_types"][car_type - 1]
    car_is_no_clunker = car_age < clunker_age_of_car_type
    price_scrapping = params.pscrap[car_type - 1]

    value_diff = (
        params.mum[consumer_type]
        * (price_scrapping - price_selling)
        / params.sigma_sell_scrapp
    )
    log_sum_diff = jnp.log(jnp.exp(value_diff) + 1)

    # Calculate the ev difference of scrapping and selling as well as the ccp of
    # scrapping
    ev_scrap_diff = (
        params.es * params.sigma_sell_scrapp * log_sum_diff * car_is_no_clunker
    )
    ccp_scrap = (
        1 - jnp.exp(-(ev_scrap_diff) / params.sigma_sell_scrapp) * car_is_no_clunker
    )

    return ccp_scrap, ev_scrap_diff, jnp.exp(value_diff)


def get_price_scrap(s_type: int, state_decision_arrays: dict, params: dict) -> float:
    """Returns the scrap price for a given state s=(s_type, s_age)."""
    assert s_type in state_decision_arrays["state_space"][:, 0]
    if s_type == 0:  # outside option
        pscrap = np.nan
    else:
        pscrap = params["pscrap"][s_type - 1]
    return pscrap


def get_expected_scrap_prices_all(
    prices, scrap_prob, state_decision_arrays, params: dict, options: dict
) -> np.ndarray:
    state_space = state_decision_arrays["state_space"]
    nS = state_space.shape[0]
    assert scrap_prob.size == nS
    Ep = np.empty(nS) * np.nan
    abar = options["max_age_of_car_types"]  # num_cartypes array

    for sidx, s in enumerate(state_space):
        s_type, s_age = s
        pscrap = get_price_scrap(s_type, state_decision_arrays, params)

        map_state_to_price_index = np.array(
            state_decision_arrays["map_state_to_price_index"]
        )  # jax -> np
        price_idx = map_state_to_price_index[s]
        assert np.isscalar(
            price_idx
        ), f"price_idx should be a scalar but is {price_idx}"

        prices_u = np.array(prices["used_car_prices"])  # !!!
        if s_age == abar[s_type - 1]:
            Ep[sidx] = pscrap
        else:
            assert (
                price_idx >= 0
            ), f"price_idx should be non-negative but is {price_idx}"
            Ep[sidx] = (
                scrap_prob[sidx] * pscrap
                + (1.0 - scrap_prob[sidx]) * prices_u[price_idx]
            )

    return Ep


def expected_scrap_price(
    sidx: int,
    didx: int,
    prices: dict,
    prob_scrap: np.ndarray,
    state_decision_arrays: dict,
    params: dict,
) -> float:
    # always zero if the person is not getting rid of a car
    number_of_car_states = state_decision_arrays["state_space"][:, 0].size - 1
    abar = state_decision_arrays["state_space"][:, 1].max()

    assert (
        prob_scrap.size == number_of_car_states
    ), f"prob_scrap has length {prob_scrap.size} but should have length abar={abar}"
    s = tuple(state_decision_arrays["state_space"][sidx, :])
    d = tuple(state_decision_arrays["decision_space"][didx, :])

    s_type, s_age = s
    d_own, d_type, d_age = d

    pscrap = get_price_scrap(s_type, state_decision_arrays, params)

    # s_type, s_age = s
    # d_own, d_type, d_age = d
    d_not_keep = d_own != 0
    s_not_outside = s != (0, 0)

    # returns indices
    # map_state_to_price_index = np.array(
    #    state_decision_arrays["map_state_to_price_index"]
    # )  # jax -> np
    map_state_to_price_index = np.array(
        state_decision_arrays["map_state_to_price_index"]
    )  # jax -> np
    price_idx = map_state_to_price_index[s]
    assert np.isscalar(price_idx), f"price_idx should be a scalar but is {price_idx}"

    prices_u = np.array(prices["used_car_prices"])  # !!!

    Isell = s_not_outside & d_not_keep
    if Isell:
        if s_age == abar:
            Ep = pscrap
        else:
            assert (
                price_idx >= 0
            ), f"price_idx should be non-negative but is {price_idx}"
            Ep = (
                prob_scrap[s_type - 1, s_age - 1] * pscrap
                + (1.0 - prob_scrap[s_type - 1, s_age - 1]) * prices_u[price_idx]
            )
    else:
        Ep = 0.0

    assert np.ndim(Ep) == 0, f"Ep should be a scalar but is {Ep}"
    return Ep


def get_price_buy_all(prices, state_decision_arrays, params):
    decisions = state_decision_arrays["decision_space"]
    nD = decisions.shape[0]

    p_buy = np.empty(nD)

    # params = namedtuple("model_parameters_to_estimate", params.keys())(**params)

    for didx in range(nD):
        p_buy[didx] = get_price_buy(didx, prices, state_decision_arrays, params)

    return p_buy


def get_price_buy(didx, prices, state_decision_arrays, params):
    # assert the prices is a dict
    assert isinstance(prices, dict), f"prices should be a dict but is {type(prices)}"
    assert "used_car_prices" in prices
    assert isinstance(state_decision_arrays, dict)

    decisions = state_decision_arrays["decision_space"]

    d_own, d_type, d_age = decisions[didx, :]
    if (d_own == 0) | (d_own == 1):
        return 0.0
    else:
        if d_age == 0:
            return prices["new_car_prices"][d_type - 1]
        else:
            p_buy = jpe_model.utility.calc_buying_costs(
                car_type=d_type,
                car_age=d_age,
                params=params,
                used_car_prices=prices["used_car_prices"],
                map_state_to_price_index=prices["used_prices_indexer"],
            )
            # jax -> float
            return float(p_buy)


def get_price_sell_all(
    prices, scrap_probabilities, state_decision_arrays, params, options
) -> np.ndarray:
    state_space = state_decision_arrays["state_space"]
    decision_space = state_decision_arrays["decision_space"]
    nS = state_space.shape[0]
    num_consumer_types = options["num_consumer_types"]

    assert (
        num_consumer_types == scrap_probabilities.shape[0]
    ), f"num_consumer_types={num_consumer_types} but scrap_probabilities.shape[0]={scrap_probabilities.shape[0]}"

    p_sell = np.empty((num_consumer_types, nS))
    for sidx in range(nS):
        s_type, s_age = state_space[sidx, :]
        if s_type == 0:  # outside option
            p_sell[:, sidx] = 0.0
        else:
            assert decision_space[1, 0] != 0, f"I thought didx=1 was keep?"  # not keep
            for tau in range(num_consumer_types):
                p_sell[tau, sidx] = get_price_sell(
                    sidx,
                    1,
                    prices,
                    scrap_probabilities,
                    state_decision_arrays,
                    params,
                    options,
                    tau=tau,
                )

    return p_sell


def get_price_sell(
    sidx: int,
    didx: int,
    prices: dict,
    scrap_probabilities,
    state_decision_arrays: dict,
    params: dict,
    options,
    tau=0,
) -> float:
    """
    Inputs:
        tau: (int) household type (needed for the scrap probability)
    """
    assert (
        "state_space" in state_decision_arrays
    ), f'state_decision_arrays should have a key "state_space" but has {state_decision_arrays.keys()}'
    state_space = state_decision_arrays["state_space"]
    decisions = state_decision_arrays["decision_space"]

    s = tuple(state_space[sidx, :])
    d = tuple(decisions[didx, :])

    s_type, s_age = s
    d_own, d_type, d_age = d

    d_keep = d_own == 0
    s_outside = s == (0, 0)
    # assert s <= nS-1, f's={s} is not a valid state'

    # prob_scrap_old = get_scrap_prob(tau, state_decision_arrays, params, prices, options)
    prob_scrap = scrap_probabilities[tau, :-1].reshape(options["num_car_types"], -1)  #

    if (s_outside) | (d_keep):
        return 0.0
    else:  # (s != s_outside) & (d != d_keep)
        assert ~s_outside  #
        assert s[0] >= 1, f"s={s}, cannot find a price for this state (d={d})"
        p = expected_scrap_price(
            sidx, didx, prices, prob_scrap, state_decision_arrays, params
        )
        assert np.ndim(p) == 0, f"p={p} is not a scalar"
        return p


def feasible_choice_all(sidx_vec, didx_vec, state_decision_arrays, params, options):
    N = sidx_vec.size
    assert (
        didx_vec.size == N
    ), f"sidx_vec and didx_vec should have the same length but have {sidx_vec.size} and {didx_vec.size}"
    state_space = np.array(state_decision_arrays["state_space"])
    decision_space = np.array(state_decision_arrays["decision_space"])
    nS = state_space.shape[0]
    nD = decision_space.shape[0]

    # only two decisions are illegal
    I_nocar = state_space[sidx_vec, 0] == 0
    I_purge = decision_space[didx_vec, 0] == 1
    I_keep = decision_space[didx_vec, 0] == 0
    # I_has_clunker = state_space[sidx_vec, 1] == options["max_age_of_car_types"][state_space[sidx_vec, 0]-1]
    assert (
        np.unique(options["max_age_of_car_types"]).size == 1
    ), "All car types must have the same maximum age for this code to work"
    abar = options["max_age_of_car_types"][0]
    I_has_clunker = state_space[sidx_vec, 1] == abar

    I_illegal1 = I_nocar & I_keep
    I_illegal2 = I_has_clunker & I_keep
    I_feasible = (~I_illegal1) & (~I_illegal2)

    return I_feasible


def feasible_choice(sidx, didx, state_decision_arrays, params, options):
    abars = options["max_age_of_car_types"]
    ncar_types = options["num_car_types"]
    clunker_cars = list(
        zip(np.arange(ncar_types) + 1, abars)
    )  # +1 since arange are 0,1,2,..., but we need ,1,2,3,...

    state_space = state_decision_arrays["state_space"]
    decisions = state_decision_arrays["decision_space"]
    d_own, d_type, d_age = decisions[didx, :]
    s_type, s_age = state_space[sidx, :]

    # s = state_space[sidx]
    d_keep = d_own == 0  # np.all([0,0,0] == decisions[didx,:])
    s_outside = np.all([0, 0] == state_space[sidx, :])
    #
    illegal1 = d_keep & s_outside  # (d == [d_keep]) & (s == s_outside)

    d_keep = np.all([0, 0, 0] == decisions[didx, :])
    s_clunker = tuple(state_space[sidx, :]) in clunker_cars

    illegal2 = d_keep & s_clunker

    illegal = illegal1 | illegal2
    return ~illegal


def binary_entropy_vec(p):
    E = np.zeros_like(p)
    I = (p > 0.0) & (p < 1.0)
    E[I] = -p[I] * np.log(p[I]) - (1.0 - p[I]) * np.log(1.0 - p[I])
    return E


def binary_entropy(p):
    if (p > 0.0) & (p < 1.0):
        E = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
    else:
        E = 0.0
    return E


def scrap_correction_all(
    scrap_probabilities, state_decision_arrays, options
):  # state_decision_arrays, params, prices, options):
    state_space = state_decision_arrays["state_space"]
    decision_space = state_decision_arrays["decision_space"]
    nS = state_space.shape[0]
    num_consumer_types = options["num_consumer_types"]

    assert (
        num_consumer_types == scrap_probabilities.shape[0]
    ), f"num_consumer_types={num_consumer_types} but scrap_probabilities.shape[0]={scrap_probabilities.shape[0]}"

    scrap_correction = np.empty((num_consumer_types, nS))
    for tau in range(num_consumer_types):
        scrap_prob_tau = scrap_probabilities[tau, :]
        scrap_correction[tau, :] = binary_entropy_vec(scrap_prob_tau)

    return scrap_correction


def scrap_correction(
    sidx: int,
    didx: int,
    state_decision_arrays: dict,
    params: dict,
    prices: dict,
    options,
    tau=0,
) -> float:
    state_space = state_decision_arrays["state_space"]
    decisions = state_decision_arrays["decision_space"]

    s = state_space[sidx, :]
    d = decisions[didx, :]
    s_type, s_age = s
    d_own, d_type, d_age = d

    prob_scrap = get_scrap_prob(tau, state_decision_arrays, params, prices, options)

    Isell = (s_type != 0) & (d_own != 0)  # s != nocar & d != keep
    if Isell:  # if getting rid of a car
        p = prob_scrap[sidx]
        if (p == 0.0) | (p == 1.0):
            correction = 0.0
        else:
            correction = binary_entropy(prob_scrap[sidx])
    else:
        correction = 0.0
    return correction * params["sigma_sell_scrapp"]
