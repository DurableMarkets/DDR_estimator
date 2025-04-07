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
    if specification["buying"] is None:
        return None, None
    
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
    if (nconsumers == 1) & (ncartypes == 1):
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
        "u_a", "car_type_{}_x_age_{}", specification, options
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

    return X, car_type_cols_flatten


def create_u_a_sq(
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
                main_df.loc[pd.IndexSlice[ntype, :, :, ncartype, :],"car_age_post_decision"].values ** 2                
            )

    return X, car_type_cols_flatten


def create_u_a_even(
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
            post_s_age = main_df.loc[pd.IndexSlice[ntype, :, :, ncartype, :],"car_age_post_decision"].values
            X.loc[
                pd.IndexSlice[ntype, :, :, ncartype, :], car_type_cols_looper[ntype][ncartype-1]
            ] = (
                car_type_dummies.loc[pd.IndexSlice[ntype, :, : , ncartype, :], f'car_type_{ncartype}'].values 
                * (1 - post_s_age % 2) * (post_s_age >= 4)            
            )
    return X, car_type_cols_flatten
