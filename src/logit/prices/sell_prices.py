import pandas as pd
import numpy as np

def get_price_sell_all(
    main_df, prices, scrap_probabilities, model_struct_arrays, params, options
) -> np.ndarray:
    # unpacking
    state_space = model_struct_arrays["state_space"]
    decision_space = model_struct_arrays["decision_space"]
    nS = state_space.shape[0]
    num_consumer_types = options["n_consumer_types"]

    psell_df = main_df.copy()
    psell_df["dum_hascar"] = psell_df["car_type_state"] != 0
    psell_df["dum_keep"] = psell_df["own_decision"] == 0
    psell_df["dum_getting_rid_of_car"] = psell_df["dum_hascar"] & ~psell_df["dum_keep"]

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
                    1, # 1 since it is a trade decision which is all that matters
                    prices,
                    scrap_probabilities,
                    model_struct_arrays,
                    params,
                    options,
                    tau=tau,
                )

    psell_df['price_sell'] = 0.0 # initialize at zero
    I = psell_df["dum_getting_rid_of_car"].values
    
    psell_df.loc[I, "price_sell"] = p_sell[
        psell_df.loc[I,:].index.get_level_values('consumer_type').values, 
        psell_df.loc[I,:].index.get_level_values('state').values]

    psell_df = psell_df.loc[:, ["price_sell"]]
    
    return psell_df

def get_price_sell(
    sidx: int,
    didx: int,
    prices: dict,
    scrap_probabilities,
    model_struct_arrays: dict,
    params: dict,
    options,
    tau=0,
) -> float:
    """
    Inputs:
        tau: (int) household type (needed for the scrap probability)
    """
    assert (
        "state_space" in model_struct_arrays
    ), f'model_struct_arrays should have a key "state_space" but has {model_struct_arrays.keys()}'
    state_space = model_struct_arrays["state_space"]
    decisions = model_struct_arrays["decision_space"]

    s = tuple(state_space[sidx, :])
    d = tuple(decisions[didx, :])

    s_type, s_age = s
    d_own, d_type, d_age = d

    d_keep = d_own == 0
    s_outside = s == (0, 0)
    # assert s <= nS-1, f's={s} is not a valid state'

    # prob_scrap_old = get_scrap_prob(tau, model_struct_arrays, params, prices, options)
    prob_scrap = scrap_probabilities[tau, :-1].reshape(options["n_car_types"], -1)  #

    if (s_outside) | (d_keep):
        return 0.0
    else:  # (s != s_outside) & (d != d_keep)
        assert ~s_outside  #
        assert s[0] >= 1, f"s={s}, cannot find a price for this state (d={d})"
        p = expected_scrap_price(
            sidx, didx, prices, prob_scrap, model_struct_arrays, params
        )
        assert np.ndim(p) == 0, f"p={p} is not a scalar"
        return p


def expected_scrap_price(
    sidx: int,
    didx: int,
    prices: dict,
    prob_scrap: np.ndarray,
    model_struct_arrays: dict,
    params: dict,
) -> float:
    # always zero if the person is not getting rid of a car
    number_of_car_states = model_struct_arrays["state_space"][:, 0].size - 1
    abar = model_struct_arrays["state_space"][:, 1].max()

    assert (
        prob_scrap.size == number_of_car_states
    ), f"prob_scrap has length {prob_scrap.size} but should have length abar={abar}"
    s = tuple(model_struct_arrays["state_space"][sidx, :])
    d = tuple(model_struct_arrays["decision_space"][didx, :])

    s_type, s_age = s
    d_own, d_type, d_age = d

    pscrap = get_price_scrap(s_type, model_struct_arrays, params)

    # s_type, s_age = s
    # d_own, d_type, d_age = d
    d_not_keep = d_own != 0
    s_not_outside = s != (0, 0)

    # returns indices
    # map_state_to_price_index = np.array(
    #    model_struct_arrays["map_state_to_price_index"]
    # )  # jax -> np
    map_state_to_price_index = np.array(
        model_struct_arrays["map_state_to_price_index"]
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


def get_price_scrap(s_type: int, model_struct_arrays: dict, params: dict) -> float:
    """Returns the scrap price for a given state s=(s_type, s_age)."""
    assert s_type in model_struct_arrays["state_space"][:, 0]
    if s_type == 0:  # outside option
        pscrap = np.nan
    else:
        pscrap = params["pscrap"][s_type - 1]
    return pscrap
