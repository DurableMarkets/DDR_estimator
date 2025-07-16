import jax.numpy as jnp
import logit.DDR_estimation.model_interface as mi
import numpy as np
import pandas as pd
from example_models.jpe_model.laws_of_motion import calc_accident_probability


def create_tab_index(state_decision_arrays, options):
    decision_space = state_decision_arrays["decision_space"]
    state_space = state_decision_arrays["state_space"]
    nconsumers = options["num_consumer_types"]

    nD = state_decision_arrays["decision_space"].shape[0]  # array of decision space
    nS = state_decision_arrays["state_space"].shape[0]  # array of state space

    index_array = np.vstack(
        [
            np.hstack(
                [
                    np.repeat(tau, repeats=state_space.shape[0]).reshape(-1, 1),
                    state_space,
                ]
            )
            for tau in range(nconsumers)
        ]
    )
    index_df = pd.DataFrame(index_array, columns=["tau", "no_car", "age"])
    index_df.set_index(["tau", "no_car", "age"], inplace=True)

    # verify ordering and construct ss index
    current_consumer_type = 0
    j = 0
    ss = np.zeros((nS * nconsumers)) + np.nan
    sss = np.arange(nS)
    for i, (c_type, s_type, s_age) in enumerate(index_df.index.values):
        if c_type != current_consumer_type:
            current_consumer_type = c_type
            j = 0
        s = state_decision_arrays["state_space"][j, :]
        ss[i] = sss[j]
        assert (s[0] == s_type) & (s[1] == s_age)
        j += 1

    dd = np.arange(nD)
    consumertypes = np.repeat(
        np.array(index_df.index.get_level_values(0).tolist()), nD
    ).reshape(-1, nD)
    tab = pd.DataFrame(consumertypes, index=ss, columns=dd)
    tab.index.name = "state"
    tab.columns.name = "decision"

    # rename variables:
    tab = tab.reset_index().melt(id_vars="state", value_name="tau")
    tab[["decision", "state"]] = tab[["decision", "state"]].astype(
        int
    )  # not sure why this gets converted to 'O'...

    tab = tab.set_index(["tau", "decision", "state"]).sort_index()

    return tab.index


def create_tab(dat, state_decision_arrays):
    """Constructs a tab from the data with the proper sorting of states and decisions
    indices."""
    nS = state_decision_arrays["state_space"].shape[0]  # array of state space
    nD = state_decision_arrays["decision_space"].shape[0]  # array of decision space

    # create index of states and decisions
    ss = np.arange(nS)
    dd = np.arange(nD)

    # verify ordering
    for i, (s_type, s_age) in enumerate(dat.index.values):
        s = state_decision_arrays["state_space"][i, :]
        assert (s[0] == s_type) & (s[1] == s_age)

    # tab: matrix of CCPs
    ccps = dat.drop("ev", axis=1).values
    tab = pd.DataFrame(ccps, index=ss, columns=dd)
    tab.index.name = "state"
    tab.columns.name = "decision"

    # rename variables:
    tab = tab.reset_index().melt(id_vars="state", value_name="ccp")
    tab["decision"] = tab["decision"].astype(
        int
    )  # not sure why this gets converted to 'O'...
    return tab


def create_state_dummy_matrix(state_decision_arrays):
    """
    Syntax: =create_state_dummy_matrix(state_decision_arrays)
    Creates a (ns * nd) x (ns) dummy matrix that is essentially ns by ns identity matrix repeated nd times
    but insuring that order of states is consistent with the order in the state space object in decision space arrays.
    """

    nS = state_decision_arrays["state_space"].shape[0]  # array of state space
    nD = state_decision_arrays["decision_space"].shape[0]  # array of decision space
    nSD = nS * nD

    # construct state dummy matrix
    state_ids = np.repeat(np.arange(nS).reshape(nS, 1), nD, axis=1).flatten(order="f")
    state_dummy_matrix = np.zeros((nSD, nS))

    for i in range(nSD):
        state_dummy_matrix[i, state_ids[i]] += 1

    return state_dummy_matrix


def create_state_transition_matrix(state_decision_arrays, params, options):
    """
    Syntax: create_state_transition_matrix(state_decision_arrays, params, options)
    Creates a (ns * nd) x (ns) state transition matrix for the model.
    """

    # calculate accident rates:
    # FIXME: what is the relationship between Sigma_sell_scrapp and acc_0? Wouldn't it usually be normalized?
    # FIXME: calc_ccps do not seem to work with a scalar (In general seems pretty convoluted what it does)
    # utils_scrap = jnp.append(params['acc_0'], 0)
    # accident_rate = calc_ccps(utils_scrap,params['sigma_sell_scrapp'])

    # dimensions
    nS = state_decision_arrays["state_space"].shape[0]  # array of state space
    nD = state_decision_arrays["decision_space"].shape[0]  # array of decision space
    nSD = nS * nD

    # I need to define the clunker states
    max_age_of_car_types = options["max_age_of_car_types"]
    assert np.all(
        [max_age_of_car_types == max_age_of_car_types[0]]
    ), "Does not work with differing max car ages"
    max_age_of_car_type = max_age_of_car_types[0]
    state_space = state_decision_arrays["state_space"]
    in_car_state = 1
    is_not_clunker = (state_space[:, 1] < max_age_of_car_type) & (
        state_space[:, 0] == in_car_state
    )

    # Construct and index of new states
    post_decision_flat = state_decision_arrays["post_decision_state_idxs"].flatten(
        order="F"
    )  # 'F' also works if post_decision_flat is a jax array. 'f' only works for numpy
    next_period_state = state_decision_arrays["trans_states_by_post_decision"][:, 1][
        post_decision_flat
    ]

    # construct iota matrix
    state_transition_matrix = np.zeros((nSD, nS))

    for i, next_state in enumerate(next_period_state):
        if is_not_clunker[next_state]:
            # Find the clunker state for the specific car.
            car_type = state_space[next_state, 0]
            car_age = state_space[post_decision_flat[i], 1]
            accident_rate = calc_accident_probability(car_type, car_age, params)[
                0
            ]  # returns both acciden rate and 1-accident_rate therefore first element
            clunker_idx = np.arange(nS)[
                (state_space[:, 0] == car_type)
                & (state_space[:, 1] == max_age_of_car_type)
            ]
            state_transition_matrix[i, next_state] = 1 - accident_rate
            state_transition_matrix[i, clunker_idx] = accident_rate
        else:
            state_transition_matrix[i, next_state] = 1

    return state_transition_matrix


def create_iota_space(state_decision_arrays, params, options):
    """
    Syntax: create_iota_space_new(state_decision_arrays, params, options)
    Creates a (ns * nd) x (ns) iota matrix for the model.
    """

    state_transition = create_state_transition_matrix(
        state_decision_arrays, params, options
    )
    state_dummy_matrix = create_state_dummy_matrix(state_decision_arrays)

    return params["disc_fac"] * state_transition - state_dummy_matrix


def create_iota_df(tab_index, state_decision_arrays, params, options):
    # Create iota matrix
    iota = create_iota_space(state_decision_arrays, params, options)

    # Create labels
    cols_ev, cols_ev_flat = create_ev_cols(state_decision_arrays, options)

    # Construct a df
    iota_df = pd.DataFrame(
        np.nan,
        index=tab_index,
        columns=cols_ev_flat,
    )

    # Set iota onto df
    for tau in range(options["num_consumer_types"]):
        iota_df.loc[pd.IndexSlice[tau, :, :], cols_ev[tau]] = iota

    # Remove infeasible states
    decision_state_pairs = iota_df.reset_index()[["decision", "state"]].values
    I_feasible = mi.feasible_choice_all(
        decision_state_pairs[:, 1],
        decision_state_pairs[:, 0],
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # dropping rows from iota_df
    iota_df = iota_df.loc[I_feasible, :]

    return iota_df


def create_iota_space_old(state_decision_arrays, params, options):
    """Creates the iota space for the model."""

    # dimensions
    nS = state_decision_arrays["state_space"].shape[0]  # array of state space
    nD = state_decision_arrays["decision_space"].shape[0]  # array of decision space
    nSD = nS * nD

    # create
    # So if there is multiple car types you are going to end up with different clunkers
    accident_rate = 0.1
    # I need to define the clunker states
    max_age_of_car_types = options["max_age_of_car_types"]
    assert np.all(
        [max_age_of_car_types == max_age_of_car_types[0]]
    ), "Does not work with differing max car ages"
    max_age_of_car_type = max_age_of_car_types[0]
    state_space = state_decision_arrays["state_space"]
    in_car_state = 1
    is_not_clunker = (state_space[:, 1] < max_age_of_car_type) & (
        state_space[:, 0] == in_car_state
    )

    # I then need to

    # Construct and index of new states
    post_decision_flat = state_decision_arrays["post_decision_state_idxs"].flatten(
        order="F"
    )  # 'F' also works if post_decision_flat is a jax array. 'f' only works for numpy
    next_period_state = state_decision_arrays["trans_states_by_post_decision"][:, 1][
        post_decision_flat
    ]

    # construct iota matrix
    alpha_beta_dummies = np.zeros((nSD, nS))
    state_ids = np.repeat(np.arange(nS).reshape(nS, 1), nD, axis=1).flatten(order="f")
    origin_state_dummies = np.zeros((nSD, nS))

    for i, next_state in enumerate(next_period_state):
        if is_not_clunker[next_state]:
            # Find the clunker state for the specific car.
            car_type = state_space[next_state, 0]
            clunker_idx = np.arange(nS)[
                (state_space[:, 0] == car_type)
                & (state_space[:, 1] == max_age_of_car_type)
            ]
            alpha_beta_dummies[i, next_state] = (1 - accident_rate) * params["disc_fac"]
            alpha_beta_dummies[i, clunker_idx] = accident_rate * params["disc_fac"]
        else:
            alpha_beta_dummies[i, next_state] = params["disc_fac"]

        origin_state_dummies[i, state_ids[i]] += -1

    iota = alpha_beta_dummies + origin_state_dummies

    return iota


def create_labels(state_decision_arrays):
    # TODO: I think this is a little dangerous. State_space is of dimension #cartypes*max_car_age + 1 (for no car).
    # So it will match the number of cars since we are removing one car type. But we need a dummy for a new car and not for an old car.
    #
    decisions = state_decision_arrays["decision_space"]
    states = state_decision_arrays["state_space"]
    iscardecision = decisions[:, 0] == 2
    car_decisions = decisions[iscardecision, ...]
    car_dummies = [
        f"car_{car_decisions[idx, 1]}_{car_decisions[idx, 2]}"
        for idx in range(car_decisions.shape[0])
    ]
    cols_ev = [
        f"ev_dums_{states[sidx, 0]}_{states[sidx, 1]}"
        for sidx in range(states.shape[0])
    ]

    return car_dummies, cols_ev


def create_ev_cols(state_decision_arrays, options):
    # TODO: I think this is a little dangerous. State_space is of dimension #cartypes*max_car_age + 1 (for no car).
    # So it will match the number of cars since we are removing one car type. But we need a dummy for a new car and not for an old car.
    #
    decisions = state_decision_arrays["decision_space"]
    states = state_decision_arrays["state_space"]
    num_consumer_types = options["num_consumer_types"]

    cols_ev = [
        [
            f"ev_dums_{states[sidx, 0]}_{states[sidx, 1]}_{tau}"
            for sidx in range(states.shape[0])
        ]
        for tau in range(num_consumer_types)
    ]

    cols_ev_flat = [item for sublist in cols_ev for item in sublist]

    return cols_ev, cols_ev_flat


# Need three functions:
# 1. create_X_matrix_data_dependent
# 2. create_X_matrix_data_independent
# 3. create_X_from_parts


def create_data_dependent_regressors(
    tab_index,
    prices,
    scrap_probabilities,
    state_decision_arrays,
    params,
    options,
    specification,
):
    nconsumers = options["num_consumer_types"]
    index_df = pd.DataFrame(index=tab_index)

    # Create pricing
    pricing, pricing_cols = create_pricing(
        index_df,
        prices,
        scrap_probabilities,
        state_decision_arrays,
        params,
        options,
        specification,
    )

    # Create scrap_correction
    scrap_correction, scrap_correction_cols = create_scrap_correction(
        index_df,
        prices,
        scrap_probabilities,
        state_decision_arrays,
        params,
        options,
        specification,
    )

    # Create data dependent regressors
    X_dep = pd.concat([pricing, scrap_correction], axis=1)

    # combine columns
    X_dep_cols = pricing_cols + scrap_correction_cols

    return X_dep, X_dep_cols


def create_pricing(
    tab_index,
    prices,
    scrap_probabilities,
    state_decision_arrays,
    params,
    options,
    specification,
):
    # unpack
    decision_space = np.array(state_decision_arrays["decision_space"])
    state_space = np.array(state_decision_arrays["state_space"])
    nconsumers = options["num_consumer_types"]

    Z = flow_utility_post_decision_df(tab_index, state_decision_arrays)
    Z = Z.reset_index()  # tau, decision, state are now columns in Z

    # create price cols
    price_cols, price_cols_flat = construct_utility_colnames(
        "mum", "price_{}_{}", specification, options
    )

    X = pd.DataFrame(
        np.nan,
        index=tab_index.index,
        columns=price_cols_flat,
    )
    # create helper variables
    Z["didx"] = Z["decision"]
    Z["sidx"] = Z["state"]
    Z["d_own"] = decision_space[Z["didx"], 0]
    Z["s_type"] = state_space[Z["sidx"], 0]
    Z["s_age"] = state_space[Z["sidx"], 1]
    Z["dum_buy"] = Z["d_own"] == 2
    Z["dum_hascar"] = Z["s_type"] != 0
    Z["dum_keep"] = Z["d_own"] == 0

    Z["dum_getting_rid_of_car"] = Z["dum_hascar"] & ~Z["dum_keep"]

    # Prices
    # buying and selling
    p_buy = mi.get_price_buy_all(prices, state_decision_arrays, params)
    p_sell = mi.get_price_sell_all(
        prices, scrap_probabilities, state_decision_arrays, params, options
    )

    Z["price_buy"] = 0.0  # initialize as zeros: useful for differences later
    Z["price_sell"] = 0.0

    I = Z["dum_buy"]
    p_buy_idx = Z.loc[I, "didx"].values.astype(int)
    Z.loc[I, "price_buy"] = p_buy[p_buy_idx]

    I = Z["dum_getting_rid_of_car"]
    p_sell_idx = Z.loc[I, ["s_type", "s_age"]].values.astype(int)
    p_sell_idx -= 1  # base 0
    Z.loc[I, "price_sell"] = p_sell[Z.loc[I, "tau"], Z.loc[I, "sidx"]]

    #
    Z = Z.set_index(["tau", "decision", "state"])
    price_ntypes, price_ncartypes = specification["mum"]
    if price_ntypes == 1:
        X[price_cols_flat[0]] = Z["price_sell"] - Z["price_buy"]
    elif price_ntypes == options["num_consumer_types"]:
        for tau, price_col in enumerate(price_cols_flat):
            idx_tau = pd.IndexSlice[tau, :, :]
            X.loc[idx_tau, price_col] = (
                Z.loc[idx_tau, "price_sell"] - Z.loc[idx_tau, "price_buy"]
            )

    I_feasible = mi.feasible_choice_all(
        Z["sidx"],
        Z["didx"],
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    X = X.loc[I_feasible, :]

    return X, price_cols_flat


def create_scrap_correction(
    tab_index,
    prices,
    scrap_probabilities,
    state_decision_arrays,
    params,
    options,
    specification,
):
    # unpack
    decision_space = state_decision_arrays["decision_space"]
    state_space = state_decision_arrays["state_space"]
    nconsumers = options["num_consumer_types"]

    Z = flow_utility_post_decision_df(tab_index, state_decision_arrays)
    Z = Z.reset_index()  # tau, decision, state are now columns in Z

    # create price cols
    scrap_cols, scrap_cols_flat = construct_utility_colnames(
        "scrap_correction", "scrap_correction_{}_{}", specification, options
    )

    X = pd.DataFrame(
        np.nan,
        index=tab_index.index,
        columns=scrap_cols_flat,
    )

    # create helper variables
    Z["didx"] = Z["decision"]
    Z["sidx"] = Z["state"]
    Z["d_own"] = decision_space[Z["didx"], 0]
    Z["s_type"] = state_space[Z["sidx"], 0]
    Z["dum_hascar"] = Z["s_type"] != 0
    Z["dum_keep"] = Z["d_own"] == 0

    Z["dum_getting_rid_of_car"] = Z["dum_hascar"] & ~Z["dum_keep"]

    I = Z["dum_getting_rid_of_car"]

    scrap_correction = mi.scrap_correction_all(
        scrap_probabilities, state_decision_arrays, options
    )

    # Initializing scrap_correction
    Z["scrap_correction"] = 0.0

    # Setting scrap_correction based on values in scrap_correction
    Z.loc[I, "scrap_correction"] = scrap_correction[Z.loc[I, "tau"], Z.loc[I, "sidx"]]

    # sets index
    Z = Z.set_index(["tau", "decision", "state"])

    nconsumers, ncartypes = specification["scrap_correction"]
    if nconsumers == 1:
        X.loc[:, scrap_cols_flat[0]] = Z["scrap_correction"]
    elif nconsumers > 1:
        for tau, scrap_col in enumerate(scrap_cols_flat):
            idx_tau = pd.IndexSlice[tau, :, :]
            X.loc[idx_tau, scrap_col] = Z.loc[idx_tau, "scrap_correction"]

    I_feasible = mi.feasible_choice_all(
        Z["sidx"],
        Z["didx"],
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    X = X.loc[I_feasible, :]

    return X, scrap_cols_flat


def create_data_dependent_regressors_old(
    tab_index, prices, scrap_probabilities, state_decision_arrays, params, options
):
    nconsumers = options["num_consumer_types"]
    index_df = pd.DataFrame(index=tab_index)
    for tau in range(nconsumers):
        # reduce index to
        tab_index_tau = index_df.loc[tau]

        X_dep_tau = create_data_dependent_regressors_tau(
            tab_index_tau,
            prices,
            scrap_probabilities,
            state_decision_arrays,
            params,
            options,
            tau,
        )
        # reindex for with tau
        X_dep_tau["tau"] = tau
        X_dep_tau = X_dep_tau.reset_index().set_index(["tau", "decision", "state"])
        regressors = X_dep_tau.columns.to_list()

        X_cols_rename = dict(
            zip(regressors, [f"{regressor}_{tau}" for regressor in regressors])
        )
        X_dep_tau.rename(columns=X_cols_rename, inplace=True)
        if tau == 0:
            X_dep = X_dep_tau
        else:
            X_dep = pd.concat([X_dep, X_dep_tau], axis=0)

    return X_dep


def create_data_dependent_regressors_tau(
    tab_index, prices, scrap_probabilities, state_decision_arrays, params, options, tau
):
    """Creates the part of the X matrix that is sample dependent (i.e. varies within a
    model structure). Note that scrap_probabilities can be consumer type specific.

    This includes the
     - scrap_correction (dependent on scrap probabilities),
     - price (dependent on scrap probabilities),

    Returns:
        X_data_dependent: a pandas dataframe with the sample dependent parts of the X matrix
        with a common idx compatible with X_data_independent.

    """
    decision_state_index = tab_index.index
    decision_state_pairs = tab_index.reset_index()[
        ["decision", "state"]
    ].values  # will be used for looping
    decision_space = np.array(state_decision_arrays["decision_space"])
    state_space = np.array(state_decision_arrays["state_space"])

    Z = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=["price", "scrap_correction"],
    )

    X = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=["price", "scrap_correction"],
    )

    # create helper variables
    didx_vec = decision_state_pairs[:, 0]
    sidx_vec = decision_state_pairs[:, 1]
    Z["didx"] = didx_vec
    Z["sidx"] = sidx_vec
    Z["d_own"] = decision_space[didx_vec, 0]
    Z["s_type"] = state_space[sidx_vec, 0]
    Z["s_age"] = state_space[sidx_vec, 1]
    Z["dum_buy"] = Z["d_own"] == 2
    Z["dum_hascar"] = Z["s_type"] != 0
    Z["dum_keep"] = Z["d_own"] == 0

    Z["dum_getting_rid_of_car"] = Z["dum_hascar"] & ~Z["dum_keep"]

    # scrappage correction
    I = Z["dum_getting_rid_of_car"]
    scrap_correction = mi.scrap_correction_all(
        tau, scrap_probabilities, state_decision_arrays
    )
    # , params, prices, options
    # )

    Z["scrap_correction"] = 0.0  # initialize
    Z.loc[I, "scrap_correction"] = scrap_correction[Z.loc[I, "sidx"]]

    # Prices
    # buying and selling
    p_buy = mi.get_price_buy_all(prices, state_decision_arrays, params)
    p_sell = mi.get_price_sell_all(
        prices, scrap_probabilities, state_decision_arrays, params, options, tau=tau
    )

    Z["price_buy"] = 0.0  # initialize as zeros: useful for differences later
    Z["price_sell"] = 0.0

    I = Z["dum_buy"]
    p_buy_idx = Z.loc[I, "didx"].values.astype(int)
    Z.loc[I, "price_buy"] = p_buy[p_buy_idx]

    I = Z["dum_getting_rid_of_car"]
    p_sell_idx = Z.loc[I, ["s_type", "s_age"]].values.astype(int)
    p_sell_idx -= 1  # base 0
    Z.loc[I, "price_sell"] = p_sell[Z.loc[I, "sidx"]]

    X["price"] = Z["price_sell"] - Z["price_buy"]
    X["scrap_correction"] = Z["scrap_correction"]

    I_feasible = mi.feasible_choice_all(
        sidx_vec,
        didx_vec,
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), ["price", "scrap_correction"]]
    return X


def create_u_0(
    tab_index,
    state_decision_arrays,
    params,
    options,
    specification,
):
    # Constructing the columns
    car_type_cols, car_type_cols_flatten = construct_utility_colnames(
        "u_0", "car_type_{}_{}", specification, options
    )

    decision_state_index = tab_index.index

    Z = flow_utility_post_decision_df(tab_index.loc[0, :, :], state_decision_arrays)

    # u_0 dummies
    car_type_dummies = pd.get_dummies(Z["post_s_type"], prefix="car_type")[
        ["car_type_{}".format(i) for i in range(1, options["num_car_types"] + 1)]
    ]  # Dropping the "no car" car type

    X = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=car_type_cols_flatten,
    )

    # Adding u_0
    if specification["u_0"] is None:
        car_type_dummies = []
    elif specification["u_0"][1] == 1:
        car_type_dummies = car_type_dummies.sum(axis=1)
    else:
        pass

    ntypes = len(car_type_cols)
    for ntype in range(0, options["num_consumer_types"]):
        if ntypes == 1:
            X.loc[
                pd.IndexSlice[ntype, :, :], car_type_cols[0]
            ] = car_type_dummies.values
        else:
            X.loc[
                pd.IndexSlice[ntype, :, :], car_type_cols[ntype]
            ] = car_type_dummies.values

    # FIXME: SUPER ANNOYING to do this here instead of just removing them directly from the index.
    decision_state_pairs = X.reset_index()[["decision", "state"]].values
    I_feasible = mi.feasible_choice_all(
        decision_state_pairs[:, 1],
        decision_state_pairs[:, 0],
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    return X, car_type_cols_flatten


def create_u_a(
    tab_index,
    state_decision_arrays,
    params,
    options,
    specification,
):
    # Constructing the columns
    car_type_cols, car_type_cols_flatten = construct_utility_colnames(
        "u_a", "car_type_{}_x_age_{}", specification, options
    )

    decision_state_index = tab_index.index

    Z = flow_utility_post_decision_df(tab_index.loc[0, :, :], state_decision_arrays)

    # u_0 dummies
    car_type_dummies = pd.get_dummies(Z["post_s_type"], prefix="car_type")[
        ["car_type_{}".format(i) for i in range(1, options["num_car_types"] + 1)]
    ]  # Dropping the "no car" car type

    car_type_dummies_x_age = car_type_dummies.values * Z["post_s_age"].values.reshape(
        -1, 1
    )

    X = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=car_type_cols_flatten,
    )
    # Adding u_a
    if specification["u_a"] is None:
        car_type_dummies_x_age = []
    elif specification["u_a"][1] == 1:
        car_type_dummies_x_age = car_type_dummies_x_age.sum(axis=1)
    else:
        pass

    ntypes = len(car_type_cols)
    for ntype in range(0, options["num_consumer_types"]):
        if ntypes == 1:
            X.loc[pd.IndexSlice[ntype, :, :], car_type_cols[0]] = car_type_dummies_x_age
        elif ntypes > 1:
            X.loc[
                pd.IndexSlice[ntype, :, :], car_type_cols[ntype]
            ] = car_type_dummies_x_age

    # FIXME: SUPER ANNOYING to do this here instead of just removing them directly from the index.
    decision_state_pairs = X.reset_index()[["decision", "state"]].values
    I_feasible = mi.feasible_choice_all(
        decision_state_pairs[:, 1],
        decision_state_pairs[:, 0],
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    return X, car_type_cols_flatten


def create_u_a_sq(
    tab_index,
    state_decision_arrays,
    params,
    options,
    specification,
):
    # Constructing the columns
    car_type_cols, car_type_cols_flatten = construct_utility_colnames(
        "u_a_sq", "car_type_{}_x_age_sq_{}", specification, options
    )

    decision_state_index = tab_index.index

    Z = flow_utility_post_decision_df(tab_index.loc[0, :, :], state_decision_arrays)

    # u_0 dummies
    car_type_dummies = pd.get_dummies(Z["post_s_type"], prefix="car_type")[
        ["car_type_{}".format(i) for i in range(1, options["num_car_types"] + 1)]
    ]  # Dropping the "no car" car type

    car_type_dummies_x_age_sq = (
        car_type_dummies.values * Z["post_s_age"].values.reshape(-1, 1) ** 2
    )

    X = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=car_type_cols_flatten,
    )

    # Adding u_a
    if specification["u_a_sq"] is None:
        car_type_dummies_x_age_sq = []
    elif specification["u_a_sq"][1] == 1:
        car_type_dummies_x_age_sq = car_type_dummies_x_age_sq.sum(axis=1)
    else:
        pass

    ntypes = len(car_type_cols)
    for ntype in range(0, options["num_consumer_types"]):
        if ntypes == 1:
            X.loc[
                pd.IndexSlice[ntype, :, :], car_type_cols[0]
            ] = car_type_dummies_x_age_sq
        elif ntypes > 1:
            X.loc[
                pd.IndexSlice[ntype, :, :], car_type_cols[ntype]
            ] = car_type_dummies_x_age_sq

    # FIXME: SUPER ANNOYING to do this here instead of just removing them directly from the index.
    decision_state_pairs = X.reset_index()[["decision", "state"]].values
    I_feasible = mi.feasible_choice_all(
        decision_state_pairs[:, 1],
        decision_state_pairs[:, 0],
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    return X, car_type_cols_flatten


def create_u_a_even(
    tab_index,
    state_decision_arrays,
    params,
    options,
    specification,
):
    # Constructing the columns
    car_type_cols, car_type_cols_flatten = construct_utility_colnames(
        "u_a_even", "car_type_{}_x_age_even_{}", specification, options
    )

    decision_state_index = tab_index.index

    Z = flow_utility_post_decision_df(tab_index.loc[0, :, :], state_decision_arrays)

    # u_0 dummies
    car_type_dummies = pd.get_dummies(Z["post_s_type"], prefix="car_type")[
        ["car_type_{}".format(i) for i in range(1, options["num_car_types"] + 1)]
    ]  # Dropping the "no car" car type

    post_s_age = Z["post_s_age"].values.reshape(-1, 1)

    # (1 - car_age % 2) * (car_age >= 4)
    car_type_dummies_x_age_even = (
        car_type_dummies.values * (1 - post_s_age % 2) * (post_s_age >= 4)
    )

    X = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=car_type_cols_flatten,
    )

    # Adding u_a
    if specification["u_a_even"] is None:
        car_type_dummies_x_age_even = []
    elif specification["u_a_even"][1] == 1:
        car_type_dummies_x_age_even = car_type_dummies_x_age_even.sum(axis=1)
    else:
        pass

    ntypes = len(car_type_cols)
    for ntype in range(0, options["num_consumer_types"]):
        if ntypes == 1:
            X.loc[
                pd.IndexSlice[ntype, :, :], car_type_cols[0]
            ] = car_type_dummies_x_age_even
        elif ntypes > 1:
            X.loc[
                pd.IndexSlice[ntype, :, :], car_type_cols[ntype]
            ] = car_type_dummies_x_age_even

    # FIXME: SUPER ANNOYING to do this here instead of just removing them directly from the index.
    decision_state_pairs = X.reset_index()[["decision", "state"]].values
    I_feasible = mi.feasible_choice_all(
        decision_state_pairs[:, 1],
        decision_state_pairs[:, 0],
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    return X, car_type_cols_flatten


def create_buying(
    tab_index,
    state_decision_arrays,
    params,
    options,
    specification,
):
    decision_space = state_decision_arrays["decision_space"]

    # Constructing the columns
    buying_cols, buying_cols_flatten = construct_utility_colnames(
        "buying", "buying_{}_{}", specification, options
    )

    decision_state_index = tab_index.index

    # Initialize X
    X = pd.DataFrame(
        np.nan,
        index=tab_index.index,
        columns=buying_cols_flatten,
    )

    decision_state_pairs_X = tab_index.reset_index()[
        ["decision", "state"]
    ].values  # will be used for looping

    # We're going to repeat for all consumer types so we only need to do it for one
    tab_index = tab_index.loc[0, :, :]

    Z = flow_utility_post_decision_df(tab_index, state_decision_arrays)

    decision_state_pairs = tab_index.reset_index()[
        ["decision", "state"]
    ].values  # will be used for looping

    didx_vec = decision_state_pairs[:, 0]
    Z["d_own"] = decision_space[didx_vec, 0]
    Z["dum_buy"] = Z["d_own"] == 2

    # Adding buying
    nconsumers, ncartypes = specification["buying"]
    if specification["buying"] is None:
        raise NotImplementedError(
            "Excluding transaction costs are not implemented yet."
        )
    elif (nconsumers == 1) & (ncartypes == 1):
        # sets the same dummies num_consumer_types many times (in the same column)
        for ntype in range(0, options["num_consumer_types"]):
            X.loc[pd.IndexSlice[ntype, :, :], buying_cols] = Z["dum_buy"].values
    elif (nconsumers > 1) & (ncartypes == 1):
        for ntype in range(0, options["num_consumer_types"]):
            X.loc[pd.IndexSlice[ntype, :, :], buying_cols[ntype]] = Z["dum_buy"].values
    else:
        raise NotImplementedError(
            "Psych transactions costs are only allowed to vary with consumer type, not car type."
        )

    # remove infeasible states
    I_feasible = mi.feasible_choice_all(
        decision_state_pairs_X[:, 1],
        decision_state_pairs_X[:, 0],
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # remove from X
    X = X.loc[np.array(I_feasible), :]

    return X, buying_cols_flatten


def flow_utility_post_decision_df(index, state_decision_arrays):
    """Creates the post decision state dataframe for the flow utility function."""
    post_decision_state_idxs = state_decision_arrays["post_decision_state_idxs"]
    post_decision_states = state_decision_arrays["post_decision_states"]
    state_decision_idxs = index.reset_index()[["state", "decision"]].values
    post_decision_state_idx = post_decision_state_idxs[
        state_decision_idxs[:, 0], state_decision_idxs[:, 1]
    ]
    post_decision_state = post_decision_states[post_decision_state_idx, :]

    Z = pd.DataFrame(
        post_decision_state,
        index=index.index,
        columns=["post_s_type", "post_s_age"],
    )

    return Z


def construct_utility_colnames(utility_type, variable_str, specification, options):
    if specification[utility_type] is not None:
        nconsumers, ncartypes = specification[utility_type]
    else:
        nconsumers = 0
        ncartypes = 0

    if (nconsumers == 0) | (ncartypes == 0):
        cols = []
    elif (nconsumers == 1) & (ncartypes == 1):
        cols = [variable_str.format("all", "all")]
    elif (nconsumers == 1) & (ncartypes > 1):
        cols = [
            [
                variable_str.format(ncartype, "all")
                for ncartype in range(1, options["num_car_types"] + 1)
            ]
        ]
    elif (nconsumers > 1) & (ncartypes == 1):
        cols = [
            [variable_str.format("all", nconsumer)]
            for nconsumer in range(0, options["num_consumer_types"])
        ]
    elif (nconsumers > 1) & (ncartypes > 1):
        cols = [
            [
                variable_str.format(ncartype, nconsumer)
                for ncartype in range(1, options["num_car_types"] + 1)
            ]
            for nconsumer in range(0, options["num_consumer_types"])
        ]
    else:
        raise ValueError(
            "Invalid specification chosen for utility type {}".format(utility_type)
        )

    if (nconsumers == 1) & (ncartypes == 1):
        cols_flat = cols
    else:
        cols_flat = [item for sublist in cols for item in sublist]

    return cols, cols_flat


def create_flow_utility(
    tab_index,
    iota,
    state_decision_arrays,
    params,
    options,
    specification,
):
    state_space = np.array(state_decision_arrays["state_space"])
    decision_space = np.array(state_decision_arrays["decision_space"])

    decision_state_index = tab_index.index

    # Create labels:
    car_dummies, cols_ev = create_labels(state_decision_arrays)

    decision_state_pairs = tab_index.reset_index()[
        ["decision", "state"]
    ].values  # will be used for looping

    Z = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=[],
    )

    # create helper variables
    didx_vec = decision_state_pairs[:, 0]
    sidx_vec = decision_state_pairs[:, 1]
    Z["didx"] = didx_vec
    Z["sidx"] = sidx_vec
    Z["d_own"] = decision_space[didx_vec, 0]
    Z["s_type"] = state_space[sidx_vec, 0]
    Z["s_age"] = state_space[sidx_vec, 1]
    Z["dum_buy"] = Z["d_own"] == 2
    Z["dum_purge"] = Z["d_own"] == 1
    Z["dum_nocar"] = Z["s_type"] == 0
    Z["dum_hascar"] = Z["s_type"] != 0
    Z["dum_keep"] = Z["d_own"] == 0

    Z["dum_getting_rid_of_car"] = Z["dum_hascar"] & ~Z["dum_keep"]

    Z["mid_state_idx"] = np.array(state_decision_arrays["post_decision_state_idxs"])[
        sidx_vec, didx_vec
    ]
    s_mid = np.array(state_decision_arrays["post_decision_states"])[
        Z["mid_state_idx"].values
    ]
    Z["s_mid_type"] = s_mid[:, 0]
    Z["s_mid_age"] = s_mid[:, 1]

    # initialize
    X = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=["buying"] + cols_ev,
    )

    # simple variables
    X.loc[:, cols_ev] = iota
    X["buying"] = Z["dum_buy"]

    # flow utility - Linear specification:
    post_decision_state_idxs = state_decision_arrays["post_decision_state_idxs"]
    post_decision_states = state_decision_arrays["post_decision_states"]

    state_decision_idxs = Z.reset_index()[["state", "decision"]].values
    post_decision_state_idx = post_decision_state_idxs[
        state_decision_idxs[:, 0], state_decision_idxs[:, 1]
    ]
    post_decision_state = post_decision_states[post_decision_state_idx, :]
    Z["post_s_type"] = post_decision_state[:, 0]
    Z["post_s_age"] = post_decision_state[:, 1]

    # u_0 dummies
    car_type_cols = [
        f"car_type_{i}" for i in range(1, options["num_car_types"] + 1)
    ]  # 1 to num_car_types
    car_type_dummies = pd.get_dummies(Z["post_s_type"], prefix="car_type")[
        car_type_cols
    ]  # Dropping the "no car" car type

    # u_a variables
    assert np.all(
        np.equal(options["max_age_of_car_types"], options["max_age_of_car_types"][0])
    ), "Varying ages across car type is not implemented"

    car_type_x_age_cols = [
        f"car_type_{i}_x_age" for i in range(1, options["num_car_types"] + 1)
    ]  # 1 to num_car_types

    car_type_dummies_x_age = car_type_dummies.values * Z["post_s_age"].values.reshape(
        -1, 1
    )

    # u_a_sq variables
    car_type_x_age_cols = [
        f"car_type_{i}_x_age_sq" for i in range(1, options["num_car_types"] + 1)
    ]  # 1 to num_car_types

    car_type_dummies_x_age_sq = (
        car_type_dummies.values * Z["post_s_age"].values.reshape(-1, 1) ** 2
    )

    # u_a_even variables
    car_type_x_age_even_cols = [
        f"car_type_{i}_x_age_even" for i in range(1, options["num_car_types"] + 1)
    ]  # 1 to num_car_types

    car_is_inspection_age = (1 - Z["post_s_age"] % 2) * (Z["post_s_age"] >= 4)
    car_type_dummies_x_age_even = (
        car_type_dummies.values * car_is_inspection_age.values.reshape(-1, 1)
    )

    # Doing some consistency checks
    assert (
        X.shape[0] == car_type_dummies.shape[0]
    ), "Something is wrong with the number of rows in X and car_type_dummies"
    assert (
        X.shape[0] == car_type_dummies_x_age.shape[0]
    ), "Something is wrong with the number of rows in X and car_type_dummies_x_age"

    # Adding u_0
    if specification["u_0"] is None:
        car_type_cols = [None]
        car_type_dummies = None
    elif specification["u_0"][1] == 1:
        car_type_dummies = car_type_dummies.sum(axis=1)
        car_type_cols = ["car_type_all"]
    else:
        pass

    # Adding into dataframe
    X[car_type_cols] = car_type_dummies

    # adding u_a
    if specification["u_a"] is None:
        car_type_x_age_cols = [None]
        car_type_dummies_x_age = None
    elif specification["u_a"][1] == 1:
        car_type_dummies_x_age = car_type_dummies_x_age.sum(axis=1)
        car_type_x_age_cols = ["car_type_all_x_age"]
    else:
        pass

    X[car_type_x_age_cols] = car_type_dummies_x_age

    # adding u_a_sq
    if specification["u_a_sq"] is None:
        car_type_x_age_sq_cols = [None]
        car_type_dummies_x_age_sq = None
    elif specification["u_a_sq"][1] == 1:
        car_type_dummies_x_age_sq = car_type_dummies_x_age_sq.sum(axis=1)
        car_type_x_age_sq_cols = ["car_type_all_x_age_sq"]
    else:
        pass

    X[car_type_x_age_sq_cols] = car_type_dummies_x_age_sq

    # adding u_a_even
    if specification["u_a_even"] is None:
        car_type_x_age_even_cols = [None]
        car_type_dummies_x_age_even = None
    elif specification["u_a_even"][1] == 1:
        car_type_dummies_x_age_even = car_type_dummies_x_age_even.sum(axis=1)
        car_type_x_age_even_cols = ["car_type_all_x_age_even"]
    else:
        pass

    X[car_type_x_age_even_cols] = car_type_dummies_x_age_even

    # Creating the linear specification
    linear_specification_cols = (
        car_type_cols
        + car_type_x_age_cols
        + car_type_x_age_sq_cols
        + car_type_x_age_even_cols
    )

    # remove infeasible state decision combinations
    I_feasible = mi.feasible_choice_all(
        sidx_vec,
        didx_vec,
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    model_specification = (
        ["price", "buying", "scrap_correction"] + linear_specification_cols + cols_ev
    )

    return X, model_specification


def create_model_specification(specification, options, state_decision_arrays):
    # mum
    _, mum_cols = construct_utility_colnames(
        "mum", "price_{}_{}", specification, options
    )

    # psych_trans_cost
    _, buying_cols = construct_utility_colnames(
        "buying", "buying_{}_{}", specification, options
    )

    # scrap_correction
    _, scrap_correction_cols = construct_utility_colnames(
        "scrap_correction", "scrap_correction_{}_{}", specification, options
    )

    # u_0
    _, u_0_cols = construct_utility_colnames(
        "u_0", "car_type_{}_{}", specification, options
    )

    # u_a
    _, u_a_cols = construct_utility_colnames(
        "u_a", "car_type_{}_x_age_{}", specification, options
    )

    # u_a_sq
    _, u_a_sq_cols = construct_utility_colnames(
        "u_a_sq", "car_type_{}_x_age_sq_{}", specification, options
    )

    # u_a_even
    _, u_a_even_cols = construct_utility_colnames(
        "u_a_even", "car_type_{}_x_age_even_{}", specification, options
    )

    # Iota
    _, ev_cols = create_ev_cols(state_decision_arrays, options)

    model_specification = (
        mum_cols
        + buying_cols
        + scrap_correction_cols
        + u_0_cols
        + u_a_cols
        + u_a_sq_cols
        + u_a_even_cols
        + ev_cols
    )

    return model_specification


def create_data_independent_regressors(
    tab_index, prices, state_decision_arrays, params, options, specification
):
    iota = create_iota_df(tab_index, state_decision_arrays, params, options)

    index_df = pd.DataFrame(index=tab_index)

    model_specification = create_model_specification(
        specification, options, state_decision_arrays
    )

    buying, _ = create_buying(
        index_df,
        state_decision_arrays,
        params,
        options,
        specification,
    )

    u_0, _ = create_u_0(
        index_df,
        state_decision_arrays,
        params,
        options,
        specification,
    )

    u_a, _ = create_u_a(
        index_df,
        state_decision_arrays,
        params,
        options,
        specification,
    )

    u_a_sq, _ = create_u_a_sq(
        index_df,
        state_decision_arrays,
        params,
        options,
        specification,
    )

    u_a_even, _ = create_u_a_even(
        index_df,
        state_decision_arrays,
        params,
        options,
        specification,
    )

    # Combine all the flow variables

    X_indep = pd.concat([buying, u_0, u_a, u_a_sq, u_a_even, iota], axis=1)

    return X_indep, model_specification


def create_data_independent_regressors_tau(
    tab_index,
    iota,
    prices,
    state_decision_arrays,
    params,
    options,
    tau,
    linear_specification,
):
    """Creates the regression variables that are sample independent ie.

    all the variables that stems from the model structure.
    Returns:
      a pandas dataframe with the sample independent parts of the X matrix.

    """

    state_space = np.array(state_decision_arrays["state_space"])
    decision_space = np.array(state_decision_arrays["decision_space"])

    decision_state_index = tab_index.index

    # Create labels:
    car_dummies, cols_ev = create_labels(state_decision_arrays)

    decision_state_pairs = tab_index.reset_index()[
        ["decision", "state"]
    ].values  # will be used for looping

    Z = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=["pbuy", "psell", "pscrap", "d_own", "s_type"],
    )

    # create helper variables
    didx_vec = decision_state_pairs[:, 0]
    sidx_vec = decision_state_pairs[:, 1]
    Z["didx"] = didx_vec
    Z["sidx"] = sidx_vec
    Z["d_own"] = decision_space[didx_vec, 0]
    Z["s_type"] = state_space[sidx_vec, 0]
    Z["s_age"] = state_space[sidx_vec, 1]
    Z["dum_buy"] = Z["d_own"] == 2
    Z["dum_purge"] = Z["d_own"] == 1
    Z["dum_nocar"] = Z["s_type"] == 0
    Z["dum_hascar"] = Z["s_type"] != 0
    Z["dum_keep"] = Z["d_own"] == 0

    Z["dum_getting_rid_of_car"] = Z["dum_hascar"] & ~Z["dum_keep"]

    Z["mid_state_idx"] = np.array(state_decision_arrays["post_decision_state_idxs"])[
        sidx_vec, didx_vec
    ]
    s_mid = np.array(state_decision_arrays["post_decision_states"])[
        Z["mid_state_idx"].values
    ]
    Z["s_mid_type"] = s_mid[:, 0]
    Z["s_mid_age"] = s_mid[:, 1]

    # initialize
    X = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=["buying"] + car_dummies + cols_ev,
    )

    # simple variables
    X.loc[:, cols_ev] = iota
    X["buying"] = Z["dum_buy"]

    # flow utility with dummy_specification
    mid_state_dummies = pd.get_dummies(
        Z["mid_state_idx"], prefix="car"
    )  # , drop_first=True, dtype=float)

    # renaming for better interpretation of explanatory variables
    car_dummies_rename = dict(zip(mid_state_dummies.columns.tolist(), car_dummies))
    mid_state_dummies = mid_state_dummies.rename(columns=car_dummies_rename)

    # ev_dummies rename
    ev_dummies_rename = dict(
        zip(cols_ev, [f"ev_{sidx}" for sidx in range(state_space.shape[0])])
    )
    mid_state_dummies = mid_state_dummies.rename(columns=ev_dummies_rename)

    cols = mid_state_dummies.columns
    X[cols] = mid_state_dummies[cols]

    # flow utility - Linear specification:
    post_decision_state_idxs = state_decision_arrays["post_decision_state_idxs"]
    post_decision_states = state_decision_arrays["post_decision_states"]

    state_decision_idxs = Z.reset_index()[["state", "decision"]].values
    post_decision_state_idx = post_decision_state_idxs[
        state_decision_idxs[:, 0], state_decision_idxs[:, 1]
    ]
    post_decision_state = post_decision_states[post_decision_state_idx, :]
    Z["post_s_type"] = post_decision_state[:, 0]
    Z["post_s_age"] = post_decision_state[:, 1]

    car_type_cols = [
        f"car_type_{i}" for i in range(1, options["num_car_types"] + 1)
    ]  # 1 to num_car_types
    car_type_dummies = pd.get_dummies(Z["post_s_type"], prefix="car_type")[
        car_type_cols
    ]  # Dropping the "no car" car type

    assert np.all(
        np.equal(options["max_age_of_car_types"], options["max_age_of_car_types"][0])
    ), "Varying ages across car type is not implemented"
    car_type_x_age_cols = [
        f"car_type_{i}_x_age" for i in range(1, options["num_car_types"] + 1)
    ]  # 1 to num_car_types
    car_type_dummies_x_age = car_type_dummies.values * Z["post_s_age"].values.reshape(
        -1, 1
    )
    assert (
        X.shape[0] == car_type_dummies.shape[0]
    ), "Something is wrong with the number of rows in X and car_type_dummies"
    X[car_type_cols] = car_type_dummies
    assert (
        X.shape[0] == car_type_dummies_x_age.shape[0]
    ), "Something is wrong with the number of rows in X and car_type_dummies_x_age"
    X[car_type_x_age_cols] = car_type_dummies_x_age

    linear_specification_cols = car_type_cols + car_type_x_age_cols

    # remove infeasible state decision combinations
    I_feasible = mi.feasible_choice_all(
        sidx_vec,
        didx_vec,
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    # Dummy specification or linear specification

    if linear_specification == True:
        X.drop(columns=car_dummies, inplace=True)
        model_specification = (
            ["price", "buying", "scrap_correction"]
            + linear_specification_cols
            + cols_ev
        )
    else:
        X.drop(columns=linear_specification_cols, inplace=True)
        model_specification = (
            ["price", "buying", "scrap_correction"] + car_dummies + cols_ev
        )

    return X, model_specification


def create_regressors_combine_parts(X_indep, X_dep, model_specification):
    """Combines the data dependent and data independent parts of the X matrix."""
    X = pd.concat([X_indep, X_dep], axis=1)
    X = X.loc[:, model_specification]
    X = X.fillna(0.0)
    return X


def create_X_matrix_from_tab(
    tab,
    iota,
    prices,
    scrap_probabilities,
    state_decision_arrays,
    params,
    options,
    tau,
    linear_specification,
):
    """Creates the X matrix for the model.

    The last column of X car_xx is a dummy for the decision to purge a car.

    """
    state_space = np.array(state_decision_arrays["state_space"])
    decisions = state_decision_arrays["decision_space"]

    decision_state_index = tab.set_index(["decision", "state"]).index

    # Create labels:
    car_dummies, cols_ev = create_labels(state_decision_arrays)

    decision_state_pairs = tab[["decision", "state"]].values  # will be used for looping
    decision_space = np.array(state_decision_arrays["decision_space"])

    Z = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=["pbuy", "psell", "pscrap", "d_own", "s_type"],
    )

    # create helper variables
    didx_vec = decision_state_pairs[:, 0]
    sidx_vec = decision_state_pairs[:, 1]
    Z["didx"] = didx_vec
    Z["sidx"] = sidx_vec
    Z["d_own"] = decision_space[didx_vec, 0]
    Z["s_type"] = state_space[sidx_vec, 0]
    Z["s_age"] = state_space[sidx_vec, 1]
    Z["dum_buy"] = Z["d_own"] == 2
    Z["dum_purge"] = Z["d_own"] == 1
    Z["dum_nocar"] = Z["s_type"] == 0
    Z["dum_hascar"] = Z["s_type"] != 0
    Z["dum_keep"] = Z["d_own"] == 0

    Z["dum_getting_rid_of_car"] = Z["dum_hascar"] & ~Z["dum_keep"]

    # buying and selling
    p_buy = mi.get_price_buy_all(prices, state_decision_arrays, params)
    p_sell = mi.get_price_sell_all(
        prices, scrap_probabilities, state_decision_arrays, params, options, tau=tau
    )
    Z["price_buy"] = 0.0  # initialize as zeros: useful for differences later
    Z["price_sell"] = 0.0

    I = Z["dum_buy"]
    p_buy_idx = Z.loc[I, "didx"].values.astype(int)
    Z.loc[I, "price_buy"] = p_buy[p_buy_idx]

    I = Z["dum_getting_rid_of_car"]
    p_sell_idx = Z.loc[I, ["s_type", "s_age"]].values.astype(int)
    p_sell_idx -= 1  # base 0
    Z.loc[I, "price_sell"] = p_sell[Z.loc[I, "sidx"]]

    # scrappage correction
    # scrap_prob = mi.get_scrap_prob(tau, state_decision_arrays, params, prices, options)
    # Z['scrap_prob'] = scrap_prob[Z['s_type']-1, Z['s_age']-1]
    scrap_correction = mi.scrap_correction_all(
        tau,
        scrap_probabilities,
        state_decision_arrays,  # params, prices, options
    )
    Z["scrap_correction"] = 0.0  # initialize
    Z.loc[I, "scrap_correction"] = scrap_correction[Z.loc[I, "sidx"]]

    Z["mid_state_idx"] = np.array(state_decision_arrays["post_decision_state_idxs"])[
        sidx_vec, didx_vec
    ]
    s_mid = np.array(state_decision_arrays["post_decision_states"])[
        Z["mid_state_idx"].values
    ]
    Z["s_mid_type"] = s_mid[:, 0]
    Z["s_mid_age"] = s_mid[:, 1]

    # initialize
    X = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=["ccp", "price", "buying", "scrap_correction"] + car_dummies + cols_ev,
    )

    # simple variables
    X["ccp"] = tab["ccp"].values
    X.loc[:, cols_ev] = iota

    X["price"] = Z["price_sell"] - Z["price_buy"]
    X["buying"] = Z["dum_buy"]
    X["scrap_correction"] = Z["scrap_correction"]

    # flow utility with dummy_specification
    mid_state_dummies = pd.get_dummies(
        Z["mid_state_idx"], prefix="car"
    )  # , drop_first=True, dtype=float)

    # renaming for better interpretation of explanatory variables
    car_dummies_rename = dict(zip(mid_state_dummies.columns.tolist(), car_dummies))
    mid_state_dummies = mid_state_dummies.rename(columns=car_dummies_rename)

    # ev_dummies rename
    ev_dummies_rename = dict(
        zip(cols_ev, [f"ev_{sidx}" for sidx in range(state_space.shape[0])])
    )
    mid_state_dummies = mid_state_dummies.rename(columns=ev_dummies_rename)

    cols = mid_state_dummies.columns
    X[cols] = mid_state_dummies[cols]

    # flow utility - Linear specification:
    post_decision_state_idxs = state_decision_arrays["post_decision_state_idxs"]
    post_decision_states = state_decision_arrays["post_decision_states"]

    state_decision_idxs = Z.reset_index()[["state", "decision"]].values
    post_decision_state_idx = post_decision_state_idxs[
        state_decision_idxs[:, 0], state_decision_idxs[:, 1]
    ]
    post_decision_state = post_decision_states[post_decision_state_idx, :]
    Z["post_s_type"] = post_decision_state[:, 0]
    Z["post_s_age"] = post_decision_state[:, 1]

    car_type_cols = [
        f"car_type_{i}" for i in range(1, options["num_car_types"] + 1)
    ]  # 1 to num_car_types
    car_type_dummies = pd.get_dummies(Z["post_s_type"], prefix="car_type")[
        car_type_cols
    ]  # Dropping the "no car" car type

    assert np.all(
        np.equal(options["max_age_of_car_types"], options["max_age_of_car_types"][0])
    ), "Varying ages across car type is not implemented"
    car_type_x_age_cols = [
        f"car_type_{i}_x_age" for i in range(1, options["num_car_types"] + 1)
    ]  # 1 to num_car_types
    car_type_dummies_x_age = car_type_dummies.values * Z["post_s_age"].values.reshape(
        -1, 1
    )
    assert (
        X.shape[0] == car_type_dummies.shape[0]
    ), "Something is wrong with the number of rows in X and car_type_dummies"
    X[car_type_cols] = car_type_dummies
    assert (
        X.shape[0] == car_type_dummies_x_age.shape[0]
    ), "Something is wrong with the number of rows in X and car_type_dummies_x_age"
    X[car_type_x_age_cols] = car_type_dummies_x_age

    linear_specification_cols = car_type_cols + car_type_x_age_cols

    # remove infeasible state decision combinations
    I_feasible = mi.feasible_choice_all(
        sidx_vec,
        didx_vec,
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    # Dummy specification or linear specification

    if linear_specification == True:
        X.drop(columns=car_dummies, inplace=True)
        model_specification = (
            ["price", "buying", "scrap_correction"]
            + linear_specification_cols
            + cols_ev
        )
    else:
        X.drop(columns=linear_specification_cols, inplace=True)
        model_specification = (
            ["price", "buying", "scrap_correction"] + car_dummies + cols_ev
        )

    return X, model_specification


def create_X_matrix_from_tab_experimental(
    tab, iota, prices, state_decision_arrays, params, options, tau
):
    """Creates the X matrix for the model."""
    state_space = np.array(state_decision_arrays["state_space"])
    decisions = state_decision_arrays["decision_space"]

    decision_state_index = tab.set_index(["decision", "state"]).index

    # Create labels:
    car_dummies, cols_ev = create_labels(state_decision_arrays)

    decision_state_pairs = tab[["decision", "state"]].values  # will be used for looping
    decision_space = np.array(state_decision_arrays["decision_space"])

    Z = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=["pbuy", "psell", "pscrap", "d_own", "s_type"],
    )

    # create helper variables
    didx_vec = decision_state_pairs[:, 0]
    sidx_vec = decision_state_pairs[:, 1]
    Z["didx"] = didx_vec
    Z["sidx"] = sidx_vec
    Z["d_own"] = decision_space[didx_vec, 0]
    Z["d_age"] = decision_space[didx_vec, 2]
    Z["s_type"] = state_space[sidx_vec, 0]
    Z["s_age"] = state_space[sidx_vec, 1]
    Z["dum_buy"] = Z["d_own"] == 2
    Z["dum_purge"] = Z["d_own"] == 1
    Z["dum_nocar"] = Z["s_type"] == 0
    Z["dum_hascar"] = Z["s_type"] != 0
    Z["dum_keep"] = Z["d_own"] == 0
    # I need some dummies for buying a new car
    Z["dum_buy_new"] = (Z["d_own"] == 2) & (Z["d_age"] == 0)
    Z["dum_buy_used"] = (Z["d_own"] == 2) & (Z["d_age"] > 0)
    Z["dum_sell_used_noclunker"] = (
        (Z["d_own"] == 2) & ((Z["s_age"] > 0) & (Z["s_age"] < 25))
    ) | ((Z["d_own"] == 1) & ((Z["s_age"] > 0) & (Z["s_age"] < 25)))
    # I need a clunker dummy
    Z["is_clunker"] = (Z["s_age"] >= 25) & (Z["s_age"] < 100)

    # mapping states and decisions to unique index values:
    map_state_to_price_index = state_decision_arrays["map_state_to_price_index"]
    Z["pidx_car_bought"] = map_state_to_price_index[
        decision_space[Z["didx"], :][:, 1], decision_space[Z["didx"], :][:, 2]
    ]
    Z["pidx_car_sold"] = map_state_to_price_index[
        state_space[Z["sidx"], :][:, 0], state_space[Z["sidx"], :][:, 1]
    ]

    # Unique map to price index for used cars that are not clunkers or new cars. if clunker or new car og no transaction is taking place then -9999
    Z["dummy_car_bought"] = (1 - Z["dum_buy_used"]) * (-9999) + (Z["dum_buy_used"]) * (
        Z["pidx_car_bought"]
    )
    Z["dummy_car_sold"] = (1 - Z["dum_sell_used_noclunker"]) * (-9999) + (
        Z["dum_sell_used_noclunker"]
    ) * (Z["pidx_car_sold"])

    pd.set_option("display.max_rows", 800)
    # Z[['idx_car_bought', 'idx_car_sold']]

    # constructing dummies
    buy_dummies = pd.get_dummies(Z["dummy_car_bought"], prefix="uprice")
    sell_dummies = pd.get_dummies(Z["dummy_car_sold"], prefix="uprice")
    cols_uprice_used = buy_dummies.columns[1:]  # remove -9999

    # Buying enter negatively while selling enters positively
    uprice_dummies = -buy_dummies[cols_uprice_used].astype(int) + sell_dummies[
        cols_uprice_used
    ].astype(int)

    # I think I can use map_state_to_price_index to go from the index in decisions to the index in prices
    # I do not need the price index, but I want to get a common index.
    # I also need something that indexes what used car is bought and sold.

    # Tasks:
    # I need to modify prices such that only new car prices are in the price vector
    # or if you scrap a car then you get the scrap value

    # Then I need to add dummies for whether I buy a used car and sell a used car.

    Z["dum_getting_rid_of_car"] = Z["dum_hascar"] & ~Z["dum_keep"]

    # buying and selling
    p_buy = mi.get_price_buy_all(prices, state_decision_arrays, params)
    p_sell = mi.get_price_sell_all(
        prices, state_decision_arrays, params, options, tau=tau
    )
    Z["price_buy"] = 0.0  # initialize as zeros: useful for differences later
    Z["price_sell"] = 0.0

    I = Z["dum_buy"]
    p_buy_idx = Z.loc[I, "didx"].values.astype(int)
    Z.loc[I, "price_buy"] = p_buy[p_buy_idx]

    I = Z["dum_getting_rid_of_car"]
    p_sell_idx = Z.loc[I, ["s_type", "s_age"]].values.astype(int)
    p_sell_idx -= 1  # base 0
    Z.loc[I, "price_sell"] = p_sell[Z.loc[I, "sidx"]]

    # scrappage correction
    # scrap_prob = mi.get_scrap_prob(tau, state_decision_arrays, params, prices, options)
    # Z['scrap_prob'] = scrap_prob[Z['s_type']-1, Z['s_age']-1]
    scrap_correction = mi.scrap_correction_all(
        tau, state_decision_arrays, params, prices, options
    )
    Z["scrap_correction"] = 0.0  # initialize
    Z.loc[I, "scrap_correction"] = scrap_correction[Z.loc[I, "sidx"]]

    Z["mid_state_idx"] = np.array(state_decision_arrays["post_decision_state_idxs"])[
        sidx_vec, didx_vec
    ]
    s_mid = np.array(state_decision_arrays["post_decision_states"])[
        Z["mid_state_idx"].values
    ]
    Z["s_mid_type"] = s_mid[:, 0]
    Z["s_mid_age"] = s_mid[:, 1]

    # initialize
    X = pd.DataFrame(
        np.nan,
        index=decision_state_index,
        columns=["ccp", "price", "buying", "scrap_correction"] + car_dummies + cols_ev,
    )

    # simple variables
    X["ccp"] = tab["ccp"].values
    X.loc[:, cols_ev] = iota

    # setting price variables:
    # Enter for all new cars and all clunker cars
    X["price"] = Z["price_sell"] * Z["is_clunker"] - Z["price_buy"] * Z["dum_buy_new"]
    X[cols_uprice_used] = uprice_dummies
    X["buying"] = Z["dum_buy"]
    X["scrap_correction"] = Z["scrap_correction"]
    mid_state_dummies = pd.get_dummies(
        Z["mid_state_idx"], prefix="car"
    )  # , drop_first=True, dtype=float)
    cols = mid_state_dummies.columns
    X[cols] = mid_state_dummies[cols]

    I_feasible = mi.feasible_choice_all(
        sidx_vec,
        didx_vec,
        state_decision_arrays=state_decision_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    return X


def create_regvars(X, model_specs):
    """Creates the regression variables for the model."""
    # Create labels:

    Y = np.log(X["ccp"].values)

    # This step explicitly leaves out the outside option. This is sensible since the flow utility of the outside option is normalized to 0.
    Xreg = X[model_specs].values.astype(float)

    return Y, Xreg


# Construct and index of new states
# post_decision_flat = state_decision_arrays["post_decision_state_idxs"].flatten(order='f')
# next_period_state = state_decision_arrays["trans_states_by_post_decision"][:, 1][post_decision_flat]


# TODO: Alot of what is done here is param specific and could be done once for all mc samples.
def create_regvars_for_tau(X, model_specification, tau: int):
    """This is used specifically for the case where we have multiple datasets for
    different types of consumers."""

    Y, Xreg = create_regvars(X, model_specification)

    model_specification_tau = [s + f"_{tau}" for s in model_specification]

    return Y, Xreg, model_specification_tau


def create_regvars_for_all_tau(dat, prices, state_decision_arrays, params, options):
    """Creates regression data for all types of consumers in the data set.

    Returns:
        Ys: list of Ys for each type of consumer
        Xregs: list of regression variables for each type of consumer
        all_param_names: list of parameter names for each type of consumer

    """
    Ys = []
    Xregs = []
    all_param_names = []

    assert (
        dat.index.unique(level=0).names[0] == "consumer_type"
    ), f' You are trying to create datasets over {dat.index.unique(level=0).names} and not over "consumer_type".'
    consumertypes = dat.index.unique(level=0).values

    for tau in consumertypes:
        dat_tau = dat.loc[tau]
        assert dat_tau.index.levels[0].name == "car_type", "index should be car_type"

        tab = create_tab(dat_tau, state_decision_arrays=state_decision_arrays)
        iota = create_iota_space(
            state_decision_arrays=state_decision_arrays, params=params, options=options
        )
        X = create_X_matrix_from_tab(
            tab, iota, prices, state_decision_arrays, params, options, tau=tau
        )

        assert (X["ccp"] == 0.0).sum() == 0, "You have ccps that are equal to zero!!!"

        Y, Xreg, param_names = create_regvars_for_tau(X, state_decision_arrays, tau=tau)
        Ys.append(Y)
        Xregs.append(Xreg)
        all_param_names.append(param_names)

    return Ys, Xregs, all_param_names, consumertypes
