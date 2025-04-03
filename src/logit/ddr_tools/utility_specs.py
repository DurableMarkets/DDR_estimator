import pandas as pd
import logit.ddr_tools.utility_helpers as utility_helpers
import numpy as np


def create_buying(
    main_df,
    model_struct_arrays,
    params,
    options,
    specification,
):
    decision_space = model_struct_arrays["decision_space"]

    # Constructing the columns
    buying_cols, buying_cols_flatten, cols_looper = utility_helpers.construct_utility_colnames(
        "buying", "buying_{}_{}", specification, options
    )
    decisions = main_df.index.get_level_values("decision").values 

    helper_df = pd.DataFrame(
        np.nan,
        index=main_df.index,
        columns=buying_cols_flatten,
    )

    X = pd.DataFrame(
        np.nan,
        index=main_df.index,
        columns=buying_cols_flatten,
    )
    helper_df["d_own"] = decision_space[decisions, 0]
    helper_df["dum_buy"] = helper_df["d_own"] == 2 # buy car


    # Adding buying
    nconsumers, ncartypes = specification["buying"]
    if specification["buying"] is None:
        raise NotImplementedError(
            "Excluding transaction costs are not implemented yet."
        )
    elif (nconsumers == 1) & (ncartypes == 1):
        X.loc[:, buying_cols] = helper_df["dum_buy"].values
    elif (nconsumers > 1) & (ncartypes == 1):
        for ntype in range(0, nconsumers):
            X.loc[pd.IndexSlice[ntype, :, :], buying_cols[ntype]] = helper_df.loc[pd.IndexSlice[ntype, :, :],'dum_buy'].values
    else:
        raise NotImplementedError(
            "transactions costs are only allowed to vary with consumer type, not car type."
        )
    return X, buying_cols_flatten

def create_u_0(
    main_df,
    model_struct_arrays,
    params,
    options,
    specification,
):
    if specification['u_0'] is None:
        return None, None
        
    # Constructing the columns
    car_type_cols, car_type_cols_flatten, car_type_cols_looper = utility_helpers.construct_utility_colnames(
        "u_0", "car_type_{}_{}", specification, options
    )

    # u_0 dummies
    car_type_dummies = pd.get_dummies(main_df["car_type_post_decision"], prefix="car_type")[
        ["car_type_{}".format(i) for i in range(1, options["n_car_types"] + 1)]
    ]  # Dropping the "no car" car type

    X = pd.DataFrame(
        np.nan,
        index=main_df.index,
        columns=car_type_cols_flatten,
    )

    for ntype in range(0, options["n_consumer_types"]):
        for ncartype in range(1, options['n_car_types']+1):
            X.loc[
                pd.IndexSlice[ntype, :, :, ncartype, :], car_type_cols_looper[ntype][ncartype-1]
            ] = (
                car_type_dummies.loc[pd.IndexSlice[ntype, :, :, ncartype, :], f'car_type_{ncartype}'].values.astype(int)
)

    return X, car_type_cols_flatten


def create_u_a(
    main_df,
    model_struct_arrays,
    params,
    options,
    specification,
):
    if specification['u_a'] is None:
        return None, None
        
    # Constructing the columns
    car_type_cols, car_type_cols_flatten, car_type_cols_looper = utility_helpers.construct_utility_colnames(
        "u_a", "car_type_{}_{}", specification, options
    )

    # u_0 dummies
    car_type_dummies = pd.get_dummies(main_df["car_type_post_decision"], prefix="car_type")[
        ["car_type_{}".format(i) for i in range(1, options["n_car_types"] + 1)]
    ]  # Dropping the "no car" car type

    X = pd.DataFrame(
        np.nan,
        index=main_df.index,
        columns=car_type_cols_flatten,
    )

    for ntype in range(0, options["n_consumer_types"]):
        for ncartype in range(1, options['n_car_types']+1):
            X.loc[
                pd.IndexSlice[ntype, :, :, ncartype, :], car_type_cols_looper[ntype][ncartype-1]
            ] = (
                car_type_dummies.loc[pd.IndexSlice[ntype, :, : , ncartype, :], f'car_type_{ncartype}'].values * 
                main_df.loc[pd.IndexSlice[ntype, :, :, ncartype, :],"car_age_post_decision"].values                
            )
    # The best solution would be if the label from the car_type_cols was controlling where to set the values.
    breakpoint()
    # always index over options object to make sure nothing is skipped
    # Then I need some solution to make sure that I do not skip anything.

    return X, car_type_cols_flatten


def create_u_a_sq(
    main_df,
    model_struct_arrays,
    params,
    options,
    specification,
):
    if specification['u_a_sq'] is None:
        return None, None

    ntypes, ncartypes = specification['u_a_sq']

    # Constructing the columns
    car_type_cols, car_type_cols_flatten = utility_helpers.construct_utility_colnames(
        "u_a_sq", "car_type_{}_x_age_sq_{}", specification, options
    )
    # u_0 dummies
    car_type_dummies = pd.get_dummies(main_df["car_type_post_decision"], prefix="car_type")[
        ["car_type_{}".format(i) for i in range(1, options["n_car_types"] + 1)]
    ]  # Dropping the "no car" car type

    X = pd.DataFrame(
        np.nan,
        index=main_df.index,
        columns=car_type_cols_flatten,
    )

    # Adding u_a_sq
    for ntype in range(0, ntypes):
        for ncartype in range(1, ncartypes+1):
            X.loc[
                pd.IndexSlice[ntype, :, :], car_type_cols[ntype][ncartype-1]
            ] = (
                car_type_dummies.loc[pd.IndexSlice[ntype, :, :], f'car_type_{ncartype}'].values * 
                main_df.loc[pd.IndexSlice[ntype, :, :],"car_age_post_decision"].values ** 2
    )
    breakpoint()
    return X, car_type_cols_flatten


def create_u_a_even(
    main_df,
    model_struct_arrays,
    params,
    options,
    specification,
):
    if specification['u_a_even'] is None:
        return None, None

    ntypes, ncartypes = specification['u_a_even']

    # Constructing the columns
    car_type_cols, car_type_cols_flatten = utility_helpers.construct_utility_colnames(
        "u_a_even", "car_type_{}_x_age_even_{}", specification, options
    )

    # u_0 dummies
    car_type_dummies = pd.get_dummies(main_df["car_type_post_decision"], prefix="car_type")[
        ["car_type_{}".format(i) for i in range(1, options["n_car_types"] + 1)]
    ]  # Dropping the "no car" car type

    X = pd.DataFrame(
        np.nan,
        index=main_df.index,
        columns=car_type_cols_flatten,
    )

    # Adding u_a
    for ntype in range(0, ntypes):
        for ncartype in range(1, ncartypes+1):
            breakpoint()
            post_s_age = main_df.loc[pd.IndexSlice[ntype, :, :],"car_age_post_decision"].values
            X.loc[
                pd.IndexSlice[ntype, :, :], car_type_cols[ntype][ncartype-1]
            ] = (
                car_type_dummies.loc[pd.IndexSlice[ntype, :, :], f'car_type_{ncartype}'].values 
                * (1 - post_s_age % 2) * (post_s_age >= 4)        
    )
    breakpoint()
    return X, car_type_cols_flatten
