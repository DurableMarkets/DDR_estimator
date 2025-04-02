import pandas as pd
import logit.ddr_tools.utility_helpers as utility_helpers
import numpy as np


def create_buying(
    tab_index,
    model_struct_arrays,
    params,
    options,
    specification,
):
    decision_space = model_struct_arrays["decision_space"]

    # Constructing the columns
    buying_cols, buying_cols_flatten = utility_helpers.construct_utility_colnames(
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

    Z = flow_utility_post_decision_df(tab_index, model_struct_arrays)

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
        model_struct_arrays=model_struct_arrays,
        params=params,
        options=options,
    )

    # remove from X
    X = X.loc[np.array(I_feasible), :]

    return X, buying_cols_flatten

def create_u_0(
    tab_index,
    model_struct_arrays,
    params,
    options,
    specification,
):
    # Constructing the columns
    car_type_cols, car_type_cols_flatten = utility_helpers.construct_utility_colnames(
        "u_0", "car_type_{}_{}", specification, options
    )

    decision_state_index = tab_index.index

    Z = flow_utility_post_decision_df(tab_index.loc[0, :, :], model_struct_arrays)

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
    for ntype in range(0, options["n_consumer_types"]):
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
        model_struct_arrays=model_struct_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    return X, car_type_cols_flatten


def create_u_a(
    tab_index,
    model_struct_arrays,
    params,
    options,
    specification,
):
    # Constructing the columns
    car_type_cols, car_type_cols_flatten = utility_helpers.construct_utility_colnames(
        "u_a", "car_type_{}_x_age_{}", specification, options
    )

    decision_state_index = tab_index.index

    Z = flow_utility_post_decision_df(tab_index.loc[0, :, :], model_struct_arrays)

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
        model_struct_arrays=model_struct_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    return X, car_type_cols_flatten


def create_u_a_sq(
    tab_index,
    model_struct_arrays,
    params,
    options,
    specification,
):
    # Constructing the columns
    car_type_cols, car_type_cols_flatten = utility_helpers.construct_utility_colnames(
        "u_a_sq", "car_type_{}_x_age_sq_{}", specification, options
    )

    decision_state_index = tab_index.index

    Z = flow_utility_post_decision_df(tab_index.loc[0, :, :], model_struct_arrays)

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
        model_struct_arrays=model_struct_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    return X, car_type_cols_flatten


def create_u_a_even(
    tab_index,
    model_struct_arrays,
    params,
    options,
    specification,
):
    # Constructing the columns
    car_type_cols, car_type_cols_flatten = utility_helpers.construct_utility_colnames(
        "u_a_even", "car_type_{}_x_age_even_{}", specification, options
    )

    decision_state_index = tab_index.index

    Z = flow_utility_post_decision_df(tab_index.loc[0, :, :], model_struct_arrays)

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
        model_struct_arrays=model_struct_arrays,
        params=params,
        options=options,
    )

    # For some reason a jax array of booleans cannot index a pandas dataframe, so I have to do this dumb conversion to numpy
    X = X.loc[np.array(I_feasible), :]

    return X, car_type_cols_flatten


def create_buying(
    tab_index,
    model_struct_arrays,
    params,
    options,
    specification,
):
    decision_space = model_struct_arrays["decision_space"]

    # Constructing the columns
    buying_cols, buying_cols_flatten = utility_helpers.construct_utility_colnames(
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

    Z = flow_utility_post_decision_df(tab_index, model_struct_arrays)

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
        model_struct_arrays=model_struct_arrays,
        params=params,
        options=options,
    )

    # remove from X
    X = X.loc[np.array(I_feasible), :]

    return X, buying_cols_flatten
